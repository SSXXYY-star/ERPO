import os
import torch
import numpy as np
from source.utils import reconstruct
from source.utils import transforms

import torch
import time
import pickle
import numpy as np

from source.models.molopt_score_model import log_sample_categorical, index_to_log_onehot
import copy
from source.utils.sascorer import compute_sa_score
from source.utils.docking import QVinaDockingTask
from source.models.molopt_score_model import ScorePosNet3D
from rdkit.Chem.QED import qed

class EGPO(torch.nn.Module):
    def __init__(self, config, ft_model, old_model=None, ref_model=None):
        super().__init__()
        self.config = config
        self.model = ft_model
        self.old_model = copy.deepcopy(self.model) if old_model == None else old_model
        self.ref_model = copy.deepcopy(self.model) if ref_model == None else ref_model
        self.sample_original = self.config.sample_original
        self.sample_rate = self.config.sample_rate
        

    def get_EGPO_loss(self, init_ligand_pos, init_ligand_v, batch_ligand, ligand_shape, orig_ligand_pos, orig_ligand_v,
                         sample_func, ligand_cum_atoms, protein_filename, point_cloud_center, ligand_center,
                         num_steps=None, center_pos_mode=None, use_mesh_data=None, use_pointcloud_data=None, 
                         use_pocket_data=None, grad_step=500, pred_bond=False, guide_stren=0,):
        result = self.old_model.sample_for_EGPO(
            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            batch_ligand=batch_ligand,
            ligand_shape=ligand_shape,
            num_steps=num_steps,
            center_pos_mode=center_pos_mode, 
            use_mesh_data=use_mesh_data,
            use_pointcloud_data=use_pointcloud_data,
            use_pocket_data=use_pocket_data,
            grad_step=grad_step,
            pred_bond=pred_bond,
            guide_stren=guide_stren
        )
        pos_traj, pos_mean_traj, pos_wo_pg_traj = result['pos_traj'], result['pos_mean_traj'], result['pos_wo_pg_traj']
        v_traj, vt_traj = result['v_traj'], result['vt_traj']

        time_seq = list(reversed(range(self.model.num_timesteps - num_steps, self.model.num_timesteps)))
        samp_time_seq = list(reversed(np.sort(np.random.choice(time_seq, int(num_steps * self.sample_rate), replace=False)).tolist()))  # 随机采样200个时间步

        if self.sample_original:
            num_graphs = batch_ligand.max().item() + 2

            original_batch = torch.zeros_like(orig_ligand_v)
            batch_ligand = torch.cat([batch_ligand, original_batch + num_graphs - 1], dim=0)
            ligand_shape = torch.cat([ligand_shape, ligand_shape[:128]])
            log_one_hot_original_v = index_to_log_onehot(orig_ligand_v, self.model.num_classes+int(self.model.v_mode=='tomask'))

            final_pos = torch.cat([pos_traj[-1], orig_ligand_pos], dim=0)
            final_v = torch.cat([v_traj[-1], orig_ligand_v],dim=0)
        else:
            num_graphs = batch_ligand.max().item() + 1
            final_pos = pos_traj[-1]
            final_v = v_traj[-1]

        if self.model.config.shape_mode is not None:
            ligand_shape = ligand_shape.view(num_graphs, -1, 3)

        ratios, v_kls, pos_kls = [], [], []
        for i in samp_time_seq:
            i_reversed = num_steps - 1 - i
            ligand_pos, ligand_v, pos_mean, vt, pos_wo_pg = pos_traj[i_reversed], v_traj[i_reversed], pos_mean_traj[i_reversed], vt_traj[i_reversed], pos_wo_pg_traj[i_reversed]
            
            if self.sample_original:
                t_orig = torch.full(size=(1,), fill_value=i, dtype=torch.long, device=orig_ligand_pos.device)
                ligand_pos_perturbed, ligand_v_perturbed, log_model_prob_q, pos_model_mean_q = self.perturb(orig_ligand_pos, orig_ligand_v, original_batch, time_step=t_orig, num_graphs=1)
                if i < 999:
                    ligand_pos_perturbed_t1, ligand_v_perturbed_t1, log_model_prob_q_t1, pos_model_mean_q_t1 = self.perturb(orig_ligand_pos, orig_ligand_v, original_batch, time_step=t_orig+1, num_graphs=1)
                else:
                    ligand_pos_perturbed_t1 = torch.randn(ligand_pos_perturbed.shape[0], 3, device=ligand_pos_perturbed.device)
                    ligand_v_perturbed_t1 = log_sample_categorical(torch.zeros(len(original_batch), self.model.num_classes)).to(ligand_v_perturbed.device)
                
                ligand_pos = torch.cat([ligand_pos, ligand_pos_perturbed_t1])  # 这里是输入，最初是是混乱状态，因此拼接t+1时刻的扰动结果
                ligand_v = torch.cat([ligand_v, ligand_v_perturbed_t1])

                pos_mean_q_t1_t_0 = self.model.q_pos_posterior(x0=orig_ligand_pos, xt=ligand_pos_perturbed_t1, t=t_orig, batch=original_batch)  # 这里得到的是q(x_t|x_t+1, x_0),是t=999的后验概率
                pos_mean = torch.cat([pos_mean, pos_mean_q_t1_t_0])

                # v_q_t1_t_0 = self.model.q_v_posterior(
                #     log_one_hot_original_v, 
                #     self.model.q_v_pred(
                #         log_one_hot_original_v, 
                #         t_orig + 1, 
                #         original_batch
                #     )  # q(v_t+1|v_0)
                #         if i < 999 else 
                #         index_to_log_onehot(
                #             ligand_v_perturbed_t1,  # 如果是最后一步，则从均匀分布采样
                #             self.model.num_classes+int(self.model.v_mode=='tomask')
                #         ), 
                #     t_orig + 1, 
                #     original_batch,
                # )  # q(v_t|v_t+1, v_0)

                v_q_t1_t_0 = self.model.q_v_posterior(
                    log_one_hot_original_v, 
                    index_to_log_onehot(
                        ligand_v_perturbed_t1,  # 如果是最后一步，则从均匀分布采样
                        self.model.num_classes+int(self.model.v_mode=='tomask')
                    ), 
                    t_orig, 
                    original_batch,
                )  
                # q(v_t|v_t+1, v_0)  
                # 以t=999为例, 此处需要求得q(v_999|v_1000, v_0), q_v_posterior中t为1000，求得的t-1为999时刻的分布, 
                # index_to_log_onehot(ligand_v_perturbed_t1)即为v_1000, 
                # perturb函数采样的t=999时刻，类比为999时刻的输出，但此处应该是t=999时刻的输入，因此用t=1000时刻的输入去计算999时刻的后验分布
                # 值得注意的是，本来的代码中就用的是t_orig而不是t_orig+1，经过确认这是正确的，因为q_v_posterior中会对t做-1处理
                vt = torch.cat([vt, v_q_t1_t_0])

                pos_wo_pg = torch.cat([pos_wo_pg, ligand_pos_perturbed])  # 这里是输出，最初是从混乱状态过了E3模型得到的，因此拼接t时刻的扰动结果

            with torch.no_grad():
                pos_model_mean_ref, ligand_pos_ref, vt_log_prob_ref, ligand_v_ref, _ = self.ref_model.get_one_step_with_grad(
                    ligand_pos, ligand_v, batch_ligand, ligand_shape, pred_bond, i, guide_stren, 
                    center_pos_mode, num_graphs
                )
                    
            pos_model_mean, ligand_pos, vt_log_prob, ligand_v, pos_variance = self.model.get_one_step_with_grad(
                ligand_pos, ligand_v, batch_ligand, ligand_shape, pred_bond, i, guide_stren, 
                center_pos_mode, num_graphs
            )
            for j in range(num_graphs):
                # 计算原子类型策略比
                prob_v, prob_v_old, prob_v_ref = vt_log_prob[batch_ligand == j], vt[batch_ligand == j], vt_log_prob_ref[batch_ligand == j]
                v_old_max, v_action = prob_v_old.max(dim=1)
                r_type = torch.exp(prob_v[torch.arange(prob_v.shape[0], device=prob_v.device), v_action] - v_old_max).mean()  # 为最大的原子类型比作为除法
                # 计算原子位置的对数概率
                # if j >= num_graphs:
                #     r_pos = self.approximate_position_ratio(pos_model_mean[new_batch_ligand == j], pos_mean_traj[i_reversed][new_batch_ligand == j], pos_variance[new_batch_ligand == j])
                # else:
                #     r_pos = self.position_ratio(pos_traj_4_ratio[i_reversed][new_batch_ligand == j], pos_model_mean[new_batch_ligand == j], pos_mean_traj[i_reversed][new_batch_ligand == j], pos_variance[new_batch_ligand == j]) # P(x_t-1|x_t) / P_old(x_t-1|x_t) # P(x_t-1|x_t) / P_old(x_t-1|x_t)
                r_pos = self.position_ratio(pos_wo_pg[batch_ligand == j], pos_model_mean[batch_ligand == j], pos_mean[batch_ligand == j], pos_variance[batch_ligand == j]) # P(x_t-1|x_t) / P_old(x_t-1|x_t) # P(x_t-1|x_t) / P_old(x_t-1|x_t)
                ratio = r_type * r_pos  # pi / pi_old
                if ratio.isinf() or ratio.isnan() or i == 0:
                    print('inf or nan ratio in ', i)
                    ratio = r_type   # todo 在这里把t=0时刻的position ratio直接设置为1。查看公式，是否能把这一项去除掉？
                pos_kl = self.position_kl(pos_model_mean[j], pos_model_mean_ref[j])
                v_kl = self.type_kl(prob_v, prob_v_ref)

                if i == samp_time_seq[0]:
                    ratios.append(ratio)  
                    v_kls.append(v_kl)
                    pos_kls.append(pos_kl)
                else:
                    ratios[j] += ratio
                    v_kls[j] += v_kl
                    pos_kls[j] += pos_kl
        ratios = [r / len(samp_time_seq) for r in ratios]
        v_kls = [v / len(samp_time_seq) for v in v_kls]
        pos_kls = [pos / len(samp_time_seq) for pos in pos_kls] 

        if self.sample_original:
            ligand_cum_atoms.append(ligand_cum_atoms[-1] + orig_ligand_v.shape[-1])
            final_pos = [final_pos[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]] for k in range(self.config.num_within_group + 1)]  # num_samples * [num_atoms_i, 3]
            final_v = [final_v[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]] for k in range(self.config.num_within_group + 1)]  # num_samples * [num_atoms_i, 3]
        else:
            final_pos = [final_pos[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]] for k in range(self.config.num_within_group)]  # num_samples * [num_atoms_i, 3]
            final_v = [final_v[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]] for k in range(self.config.num_within_group)]  # num_samples * [num_atoms_i, 3]
        reward = self.reward_fun(
            final_pos, 
            final_v, 
            use_pocket_data,
            os.path.join(self.config.path, protein_filename[0]), 
            point_cloud_center[:3],
            ligand_center[:3],
        )

        reward = reward.view(-1)
        ratio = torch.stack(ratios)
        pos_kl = torch.stack(pos_kls)
        v_kl = torch.stack(v_kls)
        adv = reward - torch.mean(reward)
        adv = adv / (torch.std(adv) + 1e-8)
        clipped_ratios = torch.clamp(ratio, 1 - self.config.epsilon_down, 1 + self.config.epsilon_up)
        if self.sample_original:
            loss = torch.min(clipped_ratios * adv, ratio * adv)
            loss[-1] = self.config.eg_alpha * loss[-1]
            loss = -torch.mean(loss - self.config.kl_beta * (pos_kl - v_kl)) # L_clip
        else: 
            loss = -torch.mean(torch.min(clipped_ratios * adv, ratio * adv) - self.config.kl_beta * (pos_kl - v_kl))  # L_clip
        
        return loss


    @torch.no_grad()
    def reward_fun(self, pred_pos, pred_v, use_pocket_data, pocket_paths, point_cloud_center=None, ligand_center=None, root_dir=None, cycle_count=None):
        ligands = self.reconstruct_mol(pred_pos, pred_v, point_cloud_center, ligand_center)
        # 1. 类药性（QED）
        # qed = calculate_qeds(ligands)
        # 2. 靶点结合能（需对接工具如AutoDock Vina） todo 测试使用vina对接
        # count = self.val_cycle_sample * cycle_count if cycle_count is not None else 0
        aff = []
        for i in range(len(ligands)):
            if ligands[i] is None:
                aff.append([-10])
                continue
            # task = QVinaDockingTask.from_generated_mol_and_pdb_path(
            #     ligands[i],
            #     protein_path=pocket_paths,
            #     ligand_center=ligand_center
            # )
            # results = task.run_sync()
            # aff.append([-results[0]['affinity']])

            try:
                task = QVinaDockingTask.from_generated_mol_and_pdb_path(
                    ligands[i],
                    protein_path=pocket_paths,
                    ligand_center=ligand_center
                )
                results = task.run_sync()
                aff.append([-results[0]['affinity']])  # 这里转换成了正值，下面的qed和sa加法是正确的
                if self.config.use_qed_in_reward:
                    qed_value = qed(ligands[i])
                    aff[-1][0] += qed_value * self.config.qed_weight  # 将QED值放大一些以匹配结合能的数量级
                if self.config.use_sa_in_reward:
                    sa_value = compute_sa_score(ligands[i])
                    aff[-1][0] += sa_value * self.config.sa_weight  # 将SA值放大一些以匹配结合能的数量级
                # if root_dir is not None:
                #     self.save_generated_mol(ligands[i], root_dir, pocket_paths[i], count)
                #     count += 1
            except Exception as e:
                print('reward error:', e)
                print('pos:', pred_pos[i])
                print('v:', pred_v[i])
                aff.append([-20])

        # aff, pos = pred_affinity(pocket_paths, ligands)  # todo pose_score 可以使用得出的分子进行打分，不使用经MC采样后的分子，即aff是对接分数，pos是没对接时的姿态分数
        # # pos = get_pos_score(pocket_paths, ligands)
        # print(aff, pos)
        # # 3. 多靶点权重平衡
        # reward = 0.5 * aff + 0.5 * pos  # todo 这里的权重可以根据实际情况调整
        return torch.tensor(aff, device=pred_pos[0].device, dtype=torch.float32)
    def approximate_position_ratio(self, mu_pi, mu_old, sigma_t):
        # r = exp((mu_pi - mu_old)^2 / (2 * sigma^2)) 这是当动作a接近旧策略的均值μ2（采样动作的典型情况），且μ1-μ2较小时的近似 由于差分分布的动作是在高斯分布中采样，逐步加噪声而来的，因此更符合这种情况
        log_r = - (mu_pi - mu_old).pow(2) / (2 * sigma_t)  # (n, )
        return torch.exp(log_r).mean()
    
    def position_ratio(self, P_old, mu_pi, mu_old, sigma_t):
        # r = exp((mu_pi - mu_old) * (P - (mu_pi - mu_old) / 2) / sigma^2)
        log_r = (mu_pi - mu_old) * (2 * P_old - mu_pi - mu_old) / (2 * sigma_t)  # (n, 3) 两个高斯分布的比值
        return torch.exp(log_r).mean()
    
    def position_kl(self, mu_pi, mu_ref):
        # D_KL(N(mu_pi, I) || N(mu_ref, I)) = ||mu_pi - mu_ref||^2 / 2
        return (mu_pi - mu_ref).pow(2).sum() / 2

    def type_kl(self, log_p, log_q):
        # D_KL(P || Q) = sum_i p_i * (log p_i - log q_i)
        p = torch.exp(log_p)  # p = exp(log_p)
        kl_elements = p * (log_p - log_q)  # p * (log_p - log_q)
        return torch.sum(kl_elements, dim=-1).sum()
    
    def reconstruct_mol(self, pred_pos, pred_v, point_cloud_center=None, ligand_center=None):
        mols = []
        for i in range(len(pred_pos)):  
            if self.config.pos_offset:
                #protein_pos = pocket_results['data'].protein_pos
                #center_pos = torch.mean(protein_pos, dim=0, keepdim=True)
                center = point_cloud_center
                pos = pred_pos[i] + center
            else:
                pos = pred_pos[i]
            # if self.config.ppo.pos_reset:
            #     center = ligand_center[i*3:i*3+3]
            #     pos = pos + center

            pred_atom_type = transforms.get_atomic_number_from_index(pred_v[i], mode=self.config.atom_enc_mode)
            # reconstruction
            
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v[i], mode=self.config.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pos, pred_atom_type, pred_aromatic,
                                                            basic_mode=self.config.basic_mode, 
                                                            covalent_factor = self.config.covalent_factor)
                mols.append(mol)
                # smiles = Chem.MolToSmiles(mol)
            except Exception as e:
                # if self.config.sample.verbose:
                #     logger.warning('Reconstruct failed')
                print('Reconstruct failed:', e)
                print('pos:', pred_pos[i])
                print('v:', pred_v[i])
                mols.append(None)
        
        return mols
    
    def perturb(self, ligand_pos, ligand_v, batch_ligand, time_step=None, num_graphs=1):
        if isinstance(time_step, int):
            time_step = torch.full(size=(num_graphs,), fill_value=time_step, dtype=torch.long, device=ligand_pos.device)
        a = self.model.alphas_cumprod.index_select(0, time_step)
        
        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        
        pos_model_mean = a_pos.sqrt() * ligand_pos

        ligand_pos_perturbed = pos_model_mean + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std

        log_ligand_v0 = index_to_log_onehot(ligand_v, self.model.num_classes)
        log_model_prob = self.model.q_v_pred(log_ligand_v0, time_step, batch_ligand)
        ligand_v_perturbed = log_sample_categorical(log_model_prob)

        return ligand_pos_perturbed, ligand_v_perturbed, log_model_prob, pos_model_mean