import torch
from torch.utils.data import Subset
from .shape_mol_dataset import ShapeMolDataset
from .shape_data import ShapeDataset
import pdb
import numpy as np
import os

def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'shapemol':
        dataset = ShapeMolDataset(config, *args, **kwargs)
        # exit(0)
    elif name == 'shape':
        dataset = ShapeDataset(config, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config and config.dataset != 'moses2':
        dataset._connect_db()
        if not os.path.exists(config.split):
            PMDM_db = torch.load('./data/crossdocked/split_by_name.pt')
            train_set, test_set = PMDM_db['train'], PMDM_db['test']
            train_set_protein, test_set_protein = [item[0] for item in train_set], [item[0] for item in test_set]
            v_train, v_test = [], []
            for idx in range(dataset.size):
                try:
                    index = test_set_protein.index(dataset[idx]['protein_filename'])
                    v_test.append(idx)
                    print('test', idx)
                    del test_set_protein[index]
                except ValueError:
                    continue
                except TypeError as e:
                    print('error:', idx)
                    continue
            for idx in range(dataset.size):
                try:
                    index = train_set_protein.index(dataset[idx]['protein_filename'])
                    v_train.append(idx)
                    print('train', idx)
                    del train_set_protein[index]
                except ValueError:
                    continue
                except TypeError as e:
                    print('error:', idx)
                    continue

            # split = torch.load(config.split)
            np.random.shuffle(v_train)
            split = {}
            split['train'] = v_train
            split['test'] = v_test
            torch.save(split, config.split)

        split = torch.load(config.split)
        subsets = {}
        
        for k, v in split.items():
            v = [idx for idx in v if idx < dataset.size]
            if k == 'train':
                # random_valid_indices = np.random.choice(v, 1000).tolist()
                # v = [idx for idx in range(dataset.size) if idx not in random_valid_indices]
                # subsets['valid'] = Subset(dataset, indices=random_valid_indices)
                subsets['valid'] = Subset(dataset, indices=v[-2:])
                v = v[config.train_beg: config.train_end]
            subsets[k] = Subset(dataset, indices=v)
        return dataset, subsets
    elif 'split' in config and config.dataset == 'moses2':
        subsets = {}
        dataset._connect_db()
        split = torch.load(config.split)
        # 没搞懂为什么要写这个循环，也没搞懂为什么pt文件里只有20w个index。为什么不能让pt里有1584652个数据然后划分呢？为什么一定要在前20w里找1k个验证集呢？pt里的valid有什么用呢？
        for k, v in split.items():
            v = [idx for idx in v if idx < dataset.size]
            if k == 'train':
                random_valid_indices = np.random.choice(v, 1000).tolist()  # 随机选1000个作为验证集
                v = [idx for idx in range(dataset.size) if idx not in random_valid_indices]
                subsets['valid'] = Subset(dataset, indices=random_valid_indices)
            else:
                continue
            subsets[k] = Subset(dataset, indices=v)
        #split = torch.load(config.split)
        #subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
