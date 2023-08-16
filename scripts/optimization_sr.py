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

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

def optimization(model, ld_kwargs, data_loader,
                 num_starting_points=100, num_gradient_steps=5000,
                 lr=1e-3, num_saved_crys=10):
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)
        _, _, z = model.encode(batch)
        z = z[:num_starting_points].detach().clone()
        z.requires_grad = True
    else:
        z = torch.randn(num_starting_points, model.hparams.hidden_dim,
                        device=model.device)
        z.requires_grad = True

    opt = Adam([z], lr=lr)
    model.freeze()

    all_crystals = []
    interval = num_gradient_steps // (num_saved_crys-1)
    for i in tqdm(range(num_gradient_steps)):
        opt.zero_grad()
        loss = -model.fc_property(z).mean()
        loss.backward()
        opt.step()

        if i % interval == 0 or i == (num_gradient_steps-1):
            crystals = model.langevin_dynamics(z, ld_kwargs)
            all_crystals.append(crystals)
    res = {k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0) for k in
            ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}
    
    return res['frac_coords'], res['atom_types'], res['lengths'], res['angles'], res['num_atoms']


def diffusion(loader, energy, uncond, step_lr, aug):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    batch = next(iter(loader)).to(energy.device)

    all_crystals = []
    for i in range(1,11):
        print(f'Optimize from T={i*100}')
        outputs, _ = energy.sample(batch, uncond, step_lr = step_lr, diff_ratio = i/10, aug = aug)
        all_crystals.append(outputs)

    res = {k: torch.cat([d[k].detach().cpu() for d in all_crystals], dim=0).unsqueeze(0) for k in
        ['frac_coords', 'atom_types', 'num_atoms', 'lattices']}

    lengths, angles = lattices_to_params_shape(res['lattices'])
    

    return res['frac_coords'], res['atom_types'], lengths, angles, res['num_atoms']


def main(args):
    # load_data if do reconstruction.
    ld_kwargs = SimpleNamespace(n_step_each=100,
                            step_lr=args.step_lr,
                            min_sigma=0,
                            save_traj=False,
                            disable_bar=False)
    model_path = Path(args.model_path)
    model, loader, cfg = load_model(
        model_path, load_data=True)
    
    if torch.cuda.is_available():
        model.to('cuda')

    
    print('Evaluate the diffusion model.')
    
    if args.model_type == 'cdvae':
        (frac_coords, atom_types, lengths, angles, num_atoms) = optimization(model, ld_kwargs, loader)

    elif args.model_type == 'diffcsp':

        uncond_path = Path(args.uncond_path)

        uncond, _, cfg = load_model(
            uncond_path, load_data=False)    

        if torch.cuda.is_available():
            uncond.to('cuda')

        (frac_coords, atom_types, lengths, angles, num_atoms) = diffusion(loader, model, uncond, args.step_lr, args.aug)

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
    parser.add_argument('--uncond_path', default='')
    parser.add_argument('--model_type', choices=['cdvae', 'diffcsp'], required=True)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--aug', default=1e-5, type=float)
    parser.add_argument('--label', default='')
    args = parser.parse_args()


    main(args)
