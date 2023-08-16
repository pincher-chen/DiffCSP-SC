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
from pyxtal.symmetry import Group
import chemparse
import numpy as np
from p_tqdm import p_map

import pdb

import os

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


class SampleDataset(Dataset):

    def __init__(self, ori_dataset, total_num, max_atom = 80):
        super().__init__()
        self.total_num = total_num
        self.max_atom = max_atom
        #self.distribution = self.get_distribution(ori_dataset)
        self.distribution = train_dist['all']
        self.num_atoms = np.random.choice(len(self.distribution), total_num, p = self.distribution)

    def get_distribution(self, ori_dataset):
        print("Calculating data distribution from training set.")
        nums = [0 for i in range(self.max_atom + 1)]
        for i in tqdm(range(len(ori_dataset))):
            n_i = ori_dataset[i].num_atoms
            if n_i <= self.max_atom:
                nums[n_i] += 1
        return np.array(nums).astype(np.float32) / sum(nums)

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):

        num_atom = self.num_atoms[index]
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
        )
        return data

def diffusion(loader, energy, uncond, step_lr, aug):

    all_crystals = []

    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        outputs, _ = energy.sample(batch, uncond, step_lr = step_lr, aug = aug)
        all_crystals.append(outputs)

    res = {k: torch.cat([d[k].detach().cpu() for d in all_crystals], dim=0).unsqueeze(0) for k in
        ['frac_coords', 'atom_types', 'num_atoms', 'lattices']}

    lengths, angles = lattices_to_params_shape(res['lattices'])
    

    return res['frac_coords'], res['atom_types'], lengths, angles, res['num_atoms']


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, loader, cfg = load_model(
        model_path, load_data=True, testing=False)

    uncond_path = Path(args.uncond_path)

    uncond, _, cfg = load_model(
        uncond_path, load_data=False)    

    if torch.cuda.is_available():
        model.to('cuda')
        uncond.to('cuda')

    train_set = loader[0].dataset
    #print(train_set)
    #exit()

    sample_set = SampleDataset(train_set, args.total_num, args.max_atom)

    sample_loader = DataLoader(sample_set, batch_size = args.batch_size)

    print('Evaluate the diffusion model.')

    start_time = time.time()
    (frac_coords, atom_types, lengths, angles, num_atoms) = diffusion(sample_loader, model, uncond, args.step_lr, args.aug)

    if args.label == '':
        gen_out_name = 'eval_opt.pt'
    else:
        gen_out_name = f'eval_opt_{args.label}.pt'

    torch.save({
        'eval_setting': args,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
    }, model_path / gen_out_name)
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--uncond_path', required=True)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--aug', default=50, type=float)
    parser.add_argument('--total_num', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--max_atom', default=80, type=int)
    parser.add_argument('--label', default='')
    args = parser.parse_args()


    main(args)
