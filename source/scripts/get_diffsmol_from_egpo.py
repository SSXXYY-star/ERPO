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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train_EGPO/sample_diff_pos0_10_pos1.e-7_0.01_6_v001_scalar128_vec32_layer8_with_pocket_guidance_no_shape_guidance.yml')
    parser.add_argument('-i', '--data_id', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs_test_PPO')
    parser.add_argument('--protein_ligand_dist', type=str, default='./data/crossdocked/protein_ligand_dist.txt')
    parser.add_argument('--egpo_ckpt', type=str, default='models/egpo_finetune_w_sa/20260108-222031/egpo_finetune_model.pt')
    parser.add_argument('--save_path', type=str, default='models/egpo_diffusion_w_sa.pt')
    args = parser.parse_args()
    
    logger = misc.get_logger('eval_diffsmol_from_egpo')
    
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

    if 'transform' in ckpt['config'].data:
        ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    else:
        ligand_atom_mode = 'full'

    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    ft_model = ScorePosNet3D(
        ckpt['config'].model,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        ligand_bond_feature_dim=len(utils_data.BOND_TYPES)

    ).to(args.device)
    ft_model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')
    
    dists = pickle.load(open("./data/MOSES2/MOSES2_training_val_shape_atomnum_dict.pkl", 'rb'))  # 数据形式: {"分子大致所占网格大小": {"分子中原子个数": "指定网格大小指定原子个数的配体数量"}}
    

    model = EGPO(config.fine_tune, ft_model=ft_model)

    model.load_state_dict(torch.load(args.egpo_ckpt, map_location=args.device))
    
    torch.save({
        'config': ckpt['config'],
        'model': model.model.state_dict()
    }, args.save_path)

    logger.info(f'Successfully save the finetuned diffusion model to {args.save_path}!')