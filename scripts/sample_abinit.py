import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset
from eval_utils import load_model, lattices_to_params_shape, get_crystals_list

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
# from pyxtal.symmetry import Group
import chemparse
import numpy as np
from p_tqdm import p_map

import os

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

train_dist = {
    'perov' : [0, 0, 0, 0, 0, 1],
    'carbon' : [0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3250697750779839,
                0.0,
                0.27795107535708424,
                0.0,
                0.15383352487276308,
                0.0,
                0.11246100804465604,
                0.0,
                0.04958134953209654,
                0.0,
                0.038745690362830404,
                0.0,
                0.019044491873255624,
                0.0,
                0.010178952552946971,
                0.0,
                0.007059596125430964,
                0.0,
                0.006074536200952225],
    'mp' : [0.0,
            0.0021742334905660377,
            0.021079009433962265,
            0.019826061320754717,
            0.15271226415094338,
            0.047132959905660375,
            0.08464770047169812,
            0.021079009433962265,
            0.07808814858490566,
            0.03434551886792453,
            0.0972877358490566,
            0.013303360849056603,
            0.09669811320754718,
            0.02155807783018868,
            0.06522700471698113,
            0.014372051886792452,
            0.06703272405660378,
            0.00972877358490566,
            0.053176591981132074,
            0.010576356132075472,
            0.08995430424528301],
#}
    'all' : [0,
             0.000338298,
             0.013214292,
             0.075558961,
             0.192020221,
             0.01778177,
             0.040554808,
             0.006953028,
             0.088090223,
             0.024545424,
             0.038417925,
             0.005494119,
             0.113878063,
             0.0031458,
             0.013534204,
             0.002262365,
             0.200241405,
             0.000947325,
             0.008960291,
             0.002363947,
             0.018970867,
             0.002303274,
             0.006358709,
             0.000759331,
             0.030319926,
             0.000565821,
             0.00278452,
             0.000539622,
             0.009743524,
             0.000998805,
             0.007701787,
             0.000315315,
             0.007972977,
             0.000484464,
             0.001308145,
             0.000216032,
             0.007855768,
             0.000375069,
             0.001735154,
             0.001166575,
             0.014360643,
             0.000224766,
             0.001889595,
             0.000173286,
             0.002907705,
             0.000538702,
             0.001150028,
             0.00010342,
             0.003750691,
             0.000127781,
             0.000787829,
             9.37673E-05,
             0.002157567,
             7.81394E-05,
             0.000938592,
             0.000112153,
             0.004629989,
             0.000223387,
             0.000912852,
             6.84869E-05,
             0.00208908,
             3.63118E-05,
             0.000437121,
             0.000172826,
             0.002518847,
             0.00014203,
             0.000586965,
             3.63118E-05,
             0.001146811,
             8.59533E-05,
             0.000398971,
             4.55047E-05,
             0.001814673,
             3.07961E-05,
             0.000263376,
             6.52694E-05,
             0.00112153,
             6.75676E-05,
             0.000409083,
             7.3543E-05,
             0.0024499]
}

def diffusion(loader, model, step_lr):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample(batch, step_lr = step_lr)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms
    )

class SampleDataset(Dataset):

    def __init__(self, dataset, total_num):
        super().__init__()
        self.total_num = total_num
        self.distribution = train_dist[dataset]
        self.num_atoms = np.random.choice(len(self.distribution), total_num, p = self.distribution)
        self.is_carbon = dataset == 'carbon'

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):

        num_atom = self.num_atoms[index]
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
        )
        if self.is_carbon:
            data.atom_types = torch.LongTensor([6] * num_atom)
        return data

def get_pymatgen(crystal_array):
    frac_coords = crystal_array['frac_coords']
    atom_types = crystal_array['atom_types']
    lengths = crystal_array['lengths']
    angles = crystal_array['angles']
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types, coords=frac_coords, coords_are_cartesian=False)
        return structure
    except:
        return None

def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, _, cfg = load_model(
        model_path, load_data=False)

    if torch.cuda.is_available():
        model.to('cuda')

    tar_dir = os.path.join(args.save_path, args.distribution)
    os.makedirs(tar_dir, exist_ok=True)

    print('Evaluate the diffusion model.')

    test_set = SampleDataset(args.distribution, args.num_evals)
    test_loader = DataLoader(test_set, batch_size = min(args.batch_size, args.num_evals))

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = diffusion(test_loader, model, args.step_lr)

    crystal_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)

    strcuture_list = p_map(get_pymatgen, crystal_list)

    for i,structure in enumerate(strcuture_list):
        tar_file = os.path.join(tar_dir, f"{args.distribution}_{i+1}.cif")
        if structure is not None:
            writer = CifWriter(structure)
            writer.write_file(tar_file)
        else:
            print(f"{i+1} Error Structure.")
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--distribution', required=True)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--step_lr', default=1e-5, type=float)

    args = parser.parse_args()


    main(args)
