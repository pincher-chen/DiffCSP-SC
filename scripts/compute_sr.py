from collections import Counter
import argparse
import os
import json

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

import sys
sys.path.append('.')

from eval_utils import (
    smact_validity, structure_validity, CompScaler, get_fp_pdist,
    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')


class Crystal(object):

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            assert self.valid
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)



class OptEval(object):

    def __init__(self, crys, num_opt=100, eval_model_name=None, model_path=None, percentiles=None):
        """
        crys is a list of length (<step_opt> * <num_opt>),
        where <num_opt> is the number of different initialization for optimizing crystals,
        and <step_opt> is the number of saved crystals for each intialzation.
        default to minimize the property.
        """
        step_opt = int(len(crys) / num_opt)
        self.crys = crys
        self.step_opt = step_opt
        self.num_opt = num_opt
        self.eval_model_name = eval_model_name
        self.model_path = Path(model_path)
        self.percentiles = percentiles

    def get_success_rate(self):
        valid_indices = np.array([c.valid for c in self.crys])
        valid_indices = valid_indices.reshape(self.step_opt, self.num_opt)
        valid_x, valid_y = valid_indices.nonzero()
        props = np.ones([self.step_opt, self.num_opt]) * np.inf
        valid_crys = [c for c in self.crys if c.valid]
        if len(valid_crys) == 0:
            sr_5, sr_10, sr_15 = 0, 0, 0
        else:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in valid_crys], model_path=self.model_path)
            percentiles = self.percentiles
            props[valid_x, valid_y] = pred_props
            best_props = props.min(axis=0)
            sr_5 = (best_props >= percentiles[0]).mean()
            sr_10 = (best_props >= percentiles[1]).mean()
            sr_15 = (best_props >= percentiles[2]).mean()
        return {'SR5': sr_5, 'SR10': sr_10, 'SR15': sr_15}

    def get_metrics(self):
        return self.get_success_rate()


def get_file_paths(root_path, task, label='', suffix='pt'):
    if args.label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    crys_array_list = get_crystals_list(
        data['frac_coords'][batch_idx],
        data['atom_types'][batch_idx],
        data['lengths'][batch_idx],
        data['angles'][batch_idx],
        data['num_atoms'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list

def get_percentiles(train_csv, prop):

    train_set = pd.read_csv(train_csv, header=0)
    train_prop = train_set[prop].tolist()
    #percentiles = [np.percentile(train_prop, i) for i in [95, 90, 85]]
    percentiles = [np.percentile(train_prop, i) for i in [90, 70, 50]]
    return percentiles



def main(args):
    all_metrics = {}

    cfg = load_config(args.root_path)
    eval_model_name = cfg.data.eval_model_name

    
    opt_file_path = get_file_paths(args.root_path, 'opt', args.label)
    crys_array_list, _ = get_crystal_array_list(opt_file_path)
    opt_crys = p_map(lambda x: Crystal(x), crys_array_list)

    percentiles = get_percentiles(train_csv=args.train_csv, prop=args.property)

    print(percentiles)


    opt_evaluator = OptEval(opt_crys, eval_model_name=eval_model_name, model_path=args.eval_model_path, percentiles=percentiles)
    opt_metrics = opt_evaluator.get_metrics()
    all_metrics = opt_metrics

    print(all_metrics)

    if args.label == '':
        metrics_out_file = 'eval_metrics.json'
    else:
        metrics_out_file = f'eval_metrics_{args.label}.json'
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--property', required=True)
    parser.add_argument('--eval_model_path', required=True)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    main(args)
