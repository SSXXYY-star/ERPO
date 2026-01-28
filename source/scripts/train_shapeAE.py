
import os
import shutil
import argparse
import numpy as np
import torch
import torch.utils.tensorboard

from torch.utils.data import DataLoader
from source.datasets import get_dataset
from source.datasets.shape_data import ShapeDataset, collate_fn
from source.models.shape_modelAE import IM_AE
from source.models.shape_pointcloud_modelAE import PointCloud_AE
from source.utils import misc as misc
from source.utils import train as utils_train
import math
from sklearn.metrics import roc_auc_score
from functools import partial

def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
    return avg_auroc / len(y_true)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/shape/se_VNDGCNN_hidden256_latent128_eclayer4_dclayer2_signeddist_pc512.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_shape_ae')
    parser.add_argument('--change_log_dir', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--continue_train_iter', type=int, default=-1)
    parser.add_argument('--train_report_iter', type=int, default=200)
    args = parser.parse_args()
    
    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging
    if args.change_log_dir is not None:
        log_dir = args.change_log_dir
    else:
        log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    if args.change_log_dir is None:
        shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
        shutil.copytree('./models', os.path.join(log_dir, 'models'))
    
    dataset, subsets = get_dataset(config=config.data)
    #print(len(subsets['test']))
    train_set, val_set = subsets['train'], subsets['valid']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')
    
    func = partial(collate_fn, config=config.data)
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=func
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False, collate_fn=func)
    
    logger.info('Building model...')
    if config.model.model_type == 'IM_AE':
        model = IM_AE(config.model).to(args.device)
    elif config.model.model_type == 'PointCloud_AE':
        model = PointCloud_AE(config.model).to(args.device)
    
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    start_iter = 1
    if args.continue_train_iter > 0 and os.path.exists(f'{ckpt_dir}/{args.continue_train_iter}.pt'):
        ckpt = torch.load(f'{ckpt_dir}/{args.continue_train_iter}.pt', map_location=args.device)
        model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
        logger.info(f'Successfully load the model! {args.continue_train_iter}.pt')
        start_iter = args.continue_train_iter + 1
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])    
    
    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))  # 用于计算参数范数，其剧烈波动可能导致优化不稳定
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))  # 用于计算梯度范数，其过大或过小可能导致梯度爆炸或消失
    
    def train(it):
        model.train()
        optimizer.zero_grad()
        batch = next(train_iterator)
        batch.to(args.device)

        loss = model.get_train_loss(
            batch.point_cloud if config.data.shape_type == 'point_cloud' else batch.voxels,
            batch.points,
            batch.values
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        
        if it % args.train_report_iter == 0:
            grad_norm_it = grad_norm(model)
            param_norm_it = param_norm(model)

            logger.info(
                '[Train] Iter %d | Loss %.6f | Lr: %.6f | Para Norm: %.6f | Grad Norm: %.6f' % (
                    it, loss, optimizer.param_groups[0]['lr'], param_norm_it, grad_norm_it
                )
            )
            writer.add_scalar(f'train/loss', loss, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', grad_norm_it, it)
            writer.add_scalar('train/param', param_norm_it, it)
            writer.flush()

    def validate(it):
        sum_loss, sum_acc, sum_rec, batch_num = 0, 0, 0, 0
        
        with torch.no_grad():
            model.eval()
            for batch in val_loader:
                batch.to(args.device)
                
                loss, acc, rec = model.get_val_loss(
                    batch.point_cloud if config.data.shape_type == 'point_cloud' else batch.voxels,
                    batch.points,
                    batch.values,
                )
                sum_acc += float(acc)
                sum_rec += float(rec)
                sum_loss += float(loss)
                batch_num += 1
        return sum_loss / batch_num, sum_acc / batch_num, sum_rec / batch_num
    
    try:
        best_loss, best_iter = None, None
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss, val_acc, val_rec = validate(it)

                if config.train.scheduler.type == 'plateau':
                    scheduler.step(val_loss)
                elif config.train.scheduler.type == 'warmup_plateau':
                    scheduler.step_ReduceLROnPlateau(val_loss)
                else:
                    scheduler.step()
                
                if best_loss is None or val_loss < best_loss:
                    if args.continue_train_iter > 0 and best_loss is None:
                        scheduler.best = val_loss
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}; val acc: {val_acc:.6f}; val rec: {val_rec:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}; val acc: {val_acc:.6f}; val rec: {val_rec:.6f}')
    except KeyboardInterrupt:
        logger.info('Terminating...')
