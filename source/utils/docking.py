import os
import subprocess
import random
import string
from easydict import EasyDict
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
# import pdb
from .reconstruct import reconstruct_from_generated
import openbabel.openbabel as ob


def adjust_protein_coordinates(pdb_file, ligand_center):
    """
    调整蛋白质坐标，使配体中心成为原点

    参数:
        pdb_file: 输入的PDB文件路径
        ligand_center: 配体中心坐标 [x, y, z]
    """
    # 创建OpenBabel转换对象
    conv = ob.OBConversion()
    conv.SetInFormat("pdb")

    # 读取蛋白质结构
    protein = ob.OBMol()
    if not conv.ReadFile(protein, pdb_file):
        raise ValueError(f"无法从 {pdb_file} 读取蛋白质结构")

    # 调整所有原子的坐标
    for atom in ob.OBMolAtomIter(protein):
        x = atom.GetX() - ligand_center[0]
        y = atom.GetY() - ligand_center[1]
        z = atom.GetZ() - ligand_center[2]
        atom.SetVector(float(x), float(y), float(z))

    # 保存调整后的结构
    conv_out = ob.OBConversion()
    conv_out.SetOutFormat("pdb")

    # 设置PDB输出选项
    conv_out.AddOption("r", conv_out.OUTOPTIONS)  # 输出TER记录

    pdb_block = conv_out.WriteString(protein)
    if not pdb_block:
        raise ValueError("无法将调整后的蛋白质结构转换为PDB字符串")
    return pdb_block


def get_random_id(length=30):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def load_pdb(path):
    with open(path, 'r') as f:
        return f.read()


def parse_qvina_outputs(docked_sdf_path):
    suppl = Chem.SDMolSupplier(docked_sdf_path)
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]
        results.append(EasyDict({
            'rdmol': mol,
            'mode_id': i,
            'affinity': float(line[0]),
            'rmsd_lb': float(line[1]),
            'rmsd_ub': float(line[2]),
        }))

    return results


class BaseDockingTask(object):

    def __init__(self, pdb_block, ligand_rdmol):
        super().__init__()
        self.pdb_block = pdb_block
        self.ligand_rdmol = ligand_rdmol

    def run(self):
        raise NotImplementedError()

    def get_results(self):
        raise NotImplementedError()


class QVinaDockingTask(BaseDockingTask):

    @classmethod
    def from_generated_data(cls, data, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()
        xyz = data.ligand_pos.clone().cpu().tolist()
        atomic_nums = data.ligand_element.clone().cpu().tolist()
        # print('atomic nums: ', atomic_nums)
        ligand_rdmol = reconstruct_from_generated(xyz, atomic_nums)
        # print('smiles: ', Chem.MolToSmiles(ligand_rdmol))
        return cls(pdb_block, ligand_rdmol, **kwargs)
    
    @classmethod
    def from_generated_mol_and_pdb_path(cls, ligand_rdmol, protein_path, ligand_center, **kwargs):
        # load original pdb
        pdb_block = adjust_protein_coordinates(ligand_center=ligand_center, pdb_file=protein_path)
        
        return cls(pdb_block, ligand_rdmol, **kwargs)
    
    @classmethod
    def from_generated_mol(cls, ligand_rdmol, ligand_filename, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        protein_fn = os.path.basename(ligand_filename).replace('.sdf', '_pocket10.pdb')  # PDBId_Chain_rec.pdb
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()
        return cls(pdb_block, ligand_rdmol, **kwargs)

    @classmethod
    def from_original_data(cls, data, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked',
                           **kwargs):
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'
        )
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()

        ligand_path = os.path.join(ligand_root, data.ligand_filename)
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(pdb_block, ligand_rdmol, **kwargs)

    def __init__(self, pdb_block, ligand_rdmol, conda_env='qvina_env', tmp_dir='./tmp', use_uff=True, center=None,
                 size_factor=1.):
        super().__init__(pdb_block, ligand_rdmol)
        self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = os.path.join(self.tmp_dir, self.receptor_id + '.pdb')
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        with open(self.receptor_path, 'w') as f:
            f.write(pdb_block)

        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)
        if use_uff:
            UFFOptimizeMolecule(ligand_rdmol)
        # print('after uff smiles: ', Chem.MolToSmiles(ligand_rdmol))
        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

    def run(self, exhaustiveness=16):
        commands = """
eval "$(conda shell.bash hook)"
conda activate {env}
cd {tmp}
# Prepare receptor (PDB->PDBQT)
/home/ubuntu/research/hdd01/sxy/mgltools_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r {receptor_id}.pdb
# Prepare ligand
obabel {ligand_id}.sdf -O{ligand_id}.pdbqt
qvina2 \
    --receptor {receptor_id}.pdbqt \
    --ligand {ligand_id}.pdbqt \
    --center_x {center_x:.4f} \
    --center_y {center_y:.4f} \
    --center_z {center_z:.4f} \
    --size_x {size_x} --size_y {size_y} --size_z {size_z} \
    --exhaustiveness {exhaust}
obabel {ligand_id}_out.pdbqt -O{ligand_id}_out.sdf -h
        """.format(
            receptor_id=self.receptor_id,
            ligand_id=self.ligand_id,
            env=self.conda_env,
            tmp=self.tmp_dir,
            exhaust=exhaustiveness,
            center_x=self.center[0],
            center_y=self.center[1],
            center_z=self.center[2],
            size_x=self.size_x,
            size_y=self.size_y,
            size_z=self.size_z
        )

        self.docked_sdf_path = os.path.join(self.tmp_dir, '%s_out.sdf' % self.ligand_id)

        self.proc = subprocess.Popen(
            '/bin/bash',
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        self.proc.stdin.write(commands.encode('utf-8'))
        self.proc.stdin.close()
        
        # return commands

    def run_sync(self):
        self.run()
        while self.get_results() is None:
            pass
        results = self.get_results()
        print('Best affinity:', results[0]['affinity'])
        return results

    def get_results(self):
        if self.proc is None:  # Not started
            return None
        elif self.proc.poll() is None:  # In progress
            return None
        else:
            if self.output is None:
                self.output = self.proc.stdout.readlines()
                self.error_output = self.proc.stderr.readlines()
                try:
                    # pdb.set_trace()
                    self.results = parse_qvina_outputs(self.docked_sdf_path)
                except:
                    print('[Error] Vina output error: %s' % self.docked_sdf_path)
                    return []
            return self.results
