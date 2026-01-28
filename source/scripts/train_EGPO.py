import os, sys
print(os.getcwd() + r'/source')
sys.path.append(os.getcwd())
import argparse
import math
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import shutil
from glob import glob
import time
import pickle
import source.utils.misc as misc
import source.utils.transforms as trans
from source.utils.shape import get_voxel_shape, get_pointcloud_from_mesh, get_pointcloud_from_mol, get_mesh, get_atom_stamp, build_point_shapeAE_model
from source.utils.reconstruct import reconstruct_from_generated
import source.utils.data as utils_data
from source.datasets import get_dataset
from functools import partial
from torch_geometric.transforms import Compose
from torch_geometric.data import Batch
from torch_scatter import scatter_sum, scatter_mean
from source.datasets.shape_mol_data import FOLLOW_BATCH
import trimesh
from sklearn.neighbors import KDTree
import numpy as np

from source.models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from source.preprocess.mose_training_val_dataset_generation import conformer_generation
from rdkit import Chem
import copy

from source.models.egpo_model import EGPO

def get_voxel_size(smiles):
    mol = conformer_generation(smiles)
    atom_stamp = get_atom_stamp(0.5, 4)  # 获得每种原子的网格信息
    voxel_size = np.sum(get_voxel_shape(mol, atom_stamp, 0.5, 11))  # 获得分子所占网格的大小
    return voxel_size

def sample_atom_nums(batch_size, atom_nums, atom_dist):
    return np.random.choice(atom_nums, batch_size, p=atom_dist).tolist()
    

def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v

def construct_dist_mat(protein_ligand_dist, atom_mode):
    tbl = Chem.GetPeriodicTable()
    if atom_mode == "add_aromatic":
        ligand_atom_indices = trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC
    else:
        ligand_atom_indices = trans.MAP_INDEX_TO_ATOM_TYPE_ONLY

    max_atom_index = max([tbl.GetAtomicNumber(atom) for atom_tuple in protein_ligand_dist for atom in atom_tuple]) + 1
    protein_ligand_dist_mat = np.ones((len(ligand_atom_indices), max_atom_index)) * 4
    ligand_atom_map = {}
    for ligand_atom_idx in ligand_atom_indices:
        ligand_atom = ligand_atom_indices[ligand_atom_idx]
        if ligand_atom[0] not in ligand_atom_map: ligand_atom_map[ligand_atom[0]] = []
        ligand_atom_map[ligand_atom[0]].append(ligand_atom_idx)
    
    for atom1, atom2 in protein_ligand_dist:
        dist = protein_ligand_dist[(atom1, atom2)]
        atom1_num, atom2_num = tbl.GetAtomicNumber(atom1), tbl.GetAtomicNumber(atom2)
        if atom1_num in ligand_atom_map:
            for ligand_atom_idx in ligand_atom_map[atom1_num]:
                protein_ligand_dist_mat[ligand_atom_idx, atom2_num] = dist
        
        if atom2_num in ligand_atom_map:
            for ligand_atom_idx in ligand_atom_map[atom2_num]:
                protein_ligand_dist_mat[ligand_atom_idx, atom1_num] = dist            
    
    return protein_ligand_dist_mat
    

def train_diffusion_egpo(config, model, opt, train_data, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='none',
                            sample_num_atoms='prior', dists=None,
                            grad_step=1000, guide_stren=0):

    if config.fine_tune.load_history and config.fine_tune.load_path != '':
        load_path = os.path.join(config.fine_tune.load_path)
        model.load_state_dict(torch.load(load_path, map_location=device))
        logger.info(f"Load quit model from {load_path}!")
    collate_exclude_keys = ['mol', 'ligand_index', 'id']
    t = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(config.fine_tune.save_path, t), exist_ok=True)
    update_old_model_step = config.fine_tune.update_old_model_step
    for i in tqdm(range(len(train_data))):

        if config.fine_tune.load_history and i < config.fine_tune.load_data:
            continue

        data = train_data[i]

        if config.sample.use_mesh:
            mesh = get_mesh(data['mol'], probe_radius=0.5)
            point_clouds = np.array(data['point_cloud'].squeeze(0))
            kdtree = KDTree(point_clouds)
            mesh = trimesh.Trimesh(mesh[0], mesh[1])
            use_mesh_data = (mesh, point_clouds, kdtree)
            config.sample.use_pointcloud = False
        else:
            use_mesh_data = None

        if config.sample.use_pointcloud:
            data['point_cloud'] = data['point_cloud'].cpu()
            atom_pos = np.array(data['ligand_pos'])
            point_clouds = get_pointcloud_from_mol(atom_pos)
            
            kdtree = KDTree(point_clouds)
            use_pointcloud_data = (point_clouds, kdtree, config.sample.use_pointcloud_radius)

        if "use_pocket" in config.sample and config.sample.use_pocket:
            data['protein_pos'] = data['protein_pos'].cpu().numpy()
            data['protein_element'] = data['protein_element'].cpu().numpy()
            kdtree = KDTree(data['protein_pos'])
            
            with open(args.protein_ligand_dist, 'r') as f:
                protein_ligand_dist = {}
                for line in f.readlines():
                    atom1, atom2, dist = line.strip().split(" ")
                    protein_ligand_dist[(atom1, atom2)] = float(dist)
                protein_ligand_tensor = construct_dist_mat(protein_ligand_dist, config.data.transform.ligand_atom_mode)
            use_pocket_data = (data['protein_pos'], data['protein_element'], kdtree, protein_ligand_tensor, config.sample.pocket_grad_step, config.sample.pocket_threshold)
            
        else:
            use_pocket_data = None
            protein_ligand_dist = None

        voxel_shape = get_voxel_size(data['ligand_smiles'])  # 获得标准配体在网格中的大小
        atom_nums = {}
        for key in dists.keys():  # 查找数据库中相似大小的配体有几个原子，数据频率是多少。
            if key < voxel_shape + 200 and key > voxel_shape - 200:
                atom_nums.update(dists[key])

        if len(atom_nums) == 0:  # 如果没有相似大小配体，就使用原本的配体原子数
            logger.info("failed to build atom dists as the molecules have shape volume %d and atom num %d" % (voxel_shape, data['mol'].GetNumAtoms()))
            if data['mol'].GetNumAtoms() >= 58:
                logger.info("molecule has too many atoms, skip finetune")
                continue
            sample_func = data['mol'].GetNumAtoms()
        else:  # 如果有，就对原子数进行轮盘赌
            atom_num_keys = list(atom_nums.keys())
            total_num = sum([atom_nums[key] for key in atom_num_keys])
            sample_atom_dist = [atom_nums[num]/total_num for num in atom_num_keys]
            sample_func = partial(sample_atom_nums, atom_nums=atom_num_keys, atom_dist = sample_atom_dist)  # 采样原子数
        
        if not config.sample.use_grad:
            config.sample.grad_lr = 0
        

        n_data = config.fine_tune.num_within_group
        
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], exclude_keys=collate_exclude_keys, follow_batch=list(FOLLOW_BATCH) + ['bound']).to(device)

        #with torch.no_grad():
        #ligand_cum_atoms = [0]
        if sample_num_atoms == 'size':
            if type(sample_func) is int:
                ligand_num_atoms = [sample_func] * n_data
            else:
                ligand_num_atoms = sample_func(n_data)
            ligand_cum_atoms = [0]  # batch 信息，指示从ligand_cum_atoms[i]开始是batch中第i个分子的原子
            for natom in ligand_num_atoms: ligand_cum_atoms.append(ligand_cum_atoms[-1] + natom)
            batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)  # 扩展ligand_cum_atoms
        elif sample_num_atoms == 'ref':
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            ligand_cum_atoms = [0] + [natom for natom in ligand_num_atoms]
        else:
            raise ValueError
        
        # init ligand pos
        all_ligand_atoms = sum(ligand_num_atoms)
        
        init_ligand_pos = torch.randn(all_ligand_atoms, 3).to(device)
        
        # init ligand v
        if pos_only:
            # init_ligand_v = F.one_hot(batch.ligand_atom_feature_full, num_classes=model.num_classes).float()
            init_ligand_v = batch.ligand_atom_feature_full
        else:
            if model.model.v_mode == 'gaussian':
                init_ligand_v = torch.randn(len(batch_ligand), model.model.num_classes).to(device)
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)
        
        orig_atom_num = int(batch.ligand_element_batch.shape[0] / n_data)
        model.train()
        opt.zero_grad()
        loss = model.get_EGPO_loss(
            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            batch_ligand=batch_ligand,
            ligand_shape=batch.shape_emb,
            orig_ligand_pos=batch.ligand_pos[:orig_atom_num],
            orig_ligand_v=batch.ligand_atom_feature_full[:orig_atom_num],
            sample_func=sample_func,
            ligand_cum_atoms=ligand_cum_atoms,
            protein_filename=batch.protein_filename,
            point_cloud_center=batch.point_cloud_center,
            ligand_center=batch.ligand_center,
            num_steps=num_steps,
            center_pos_mode=center_pos_mode,
            grad_step=grad_step,
            use_mesh_data=use_mesh_data,
            use_pointcloud_data=use_pointcloud_data,
            use_pocket_data=use_pocket_data,
            guide_stren=guide_stren,
            pred_bond=True,
        )

        if (i + 1) % update_old_model_step == 0:
            model.old_model = copy.deepcopy(model.model)

        loss.backward()
        opt.step()

        torch.save(model.state_dict(), os.path.join(config.fine_tune.save_path, t, 'egpo_finetune_model.pt'))
        if i % 1000 == 999:
            torch.save(model.state_dict(), os.path.join(config.fine_tune.save_path, t, f'egpo_finetune_model_step{i+1}.pt'))
        if i % config.fine_tune.upload_ref == 0:
            model.ref_model = copy.deepcopy(model.model)
    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train_EGPO/sample_diff_pos0_10_pos1.e-7_0.01_6_v001_scalar128_vec32_layer8_with_pocket_guidance_no_shape_guidance.yml')
    parser.add_argument('-i', '--data_id', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    # parser.add_argument('--result_path', type=str, default='./outputs_test_PPO')
    parser.add_argument('--protein_ligand_dist', type=str, default='./data/crossdocked/protein_ligand_dist.txt')
    args = parser.parse_args()
    
    #tmp_path = os.path.join(args.result_path, f'result_{args.data_id}.pt')
    #if os.path.exists(tmp_path): sys.exit(0)

    logger = misc.get_logger('evaluate')
    
    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    if 'train_config' in config.model:
        logger.info(f"Load training config from: {config.model['train_config']}")
        ckpt['config'] = misc.load_config(config.model['train_config'])
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    if 'transform' in ckpt['config'].data:
        ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    else:
        ligand_atom_mode = 'full'

    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # Load dataset
    dataset, subset = get_dataset(
        config=config.data,
        transform=transform
    )
    train_set, test_set = subset['train'], subset['test']
    ft_model = ScorePosNet3D(
        ckpt['config'].model,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        ligand_bond_feature_dim=len(utils_data.BOND_TYPES)

    ).to(args.device)
    ft_model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')
    
    dists = pickle.load(open("./data/MOSES2/MOSES2_training_val_shape_atomnum_dict.pkl", 'rb'))  # 数据形式: {"分子大致所占网格大小": {"分子中原子个数": "指定网格大小指定原子个数的配体数量"}}
    

    model = EGPO(config.fine_tune, ft_model=ft_model)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.fine_tune.lr, 
        amsgrad=config.fine_tune.amsgrad,
        weight_decay=config.fine_tune.weight_decay
    )

    model = train_diffusion_egpo(
        config, model, opt, train_set, 
        device=args.device,
        num_steps=config.sample.num_steps,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        guide_stren=config.sample.guide_stren,
        dists=dists
    )
