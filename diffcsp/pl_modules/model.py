from typing import Any, Dict

import hydra
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc,radius_graph_pbc,repeat_blocks)
MAX_ATOMIC_NUM = 100

from diffcsp.pl_modules.transformer import model_entrypoint

import pdb


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


class CrystGNN_Supervise(BaseModule):
    """
    GNN model for fitting the supervised objectives for crystals.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = hydra.utils.instantiate(self.hparams.encoder)

        self.task = self.hparams.task
       
        #node2graph = batch.batch
        #exit()
        #lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        #print(lattices,lattices.device)
        #exit()
        #edges_index, edge_vec = self.gen_edges(batch.num_atoms, batch.frac_coords, lattices, node2graph, edge_style='knn')
        create_model = model_entrypoint('graph_attention_transformer_nonlinear_l2_e3')
        self.trans_model = create_model(irreps_in='100x0e',radius=8,num_basis=128,out_channels=1,atomref=None,drop_path=0.0)

    def nanmean(self, input_tensor):
        mask = torch.isnan(input_tensor)
        input_tensor = torch.where(mask, torch.zeros_like(input_tensor), input_tensor)
        sum_tensor = torch.sum(input_tensor)
        count_tensor = torch.sum(~mask).type(input_tensor.dtype)
        nanmean_tensor = sum_tensor / count_tensor

        return nanmean_tensor

    def pearson_correlation(self, preds, labels):

        # Compute the mean of predictions and labels
        preds_mean = torch.mean(preds)
        labels_mean = torch.mean(labels)

        # Compute the deviations from the mean
        preds_dev = preds - preds_mean
        labels_dev = labels - labels_mean

        # Compute the covariance
        covariance = torch.sum(preds_dev * labels_dev)

        # Compute the standard deviations
        preds_std = torch.sqrt(torch.sum(preds_dev**2))
        labels_std = torch.sqrt(torch.sum(labels_dev**2))

        # Compute the Pearson correlation coefficient
        pearson_corr = covariance / (preds_std * labels_std)

        return pearson_corr

    def get_loss(self, preds, labels):

        if self.task == 'regression':
            loss = F.huber_loss(preds, labels, reduction='none')
        
        elif self.task == 'classification':
            loss = F.cross_entropy(preds, labels.reshape(-1), reduction='none')

        return loss
  
    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered   

    def reorder_symmetric_edges(
        self, edge_index, cell_offsets, neighbors, edge_vector
    ):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(
            batch_edge, minlength=neighbors.size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_vector_new,
        )

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph, edge_style):

        #if self.edge_style == 'fc':
        if edge_style == 'fc':
            lis = [torch.ones(n,n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]])
        #elif self.edge_style == 'knn':
        elif edge_style == 'knn':
            lattice_nodes = lattices[node2graph]
            cart_coords = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)

            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords, None, None, num_atoms, 8, 32,
                device=num_atoms.device, lattices=lattices)

            j_index, i_index = edge_index
            #distance_vectors = frac_coords[j_index] - frac_coords[i_index]
            distance_vectors = cart_coords[j_index] - cart_coords[i_index]
            distance_vectors += to_jimages.float()

            edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(edge_index, to_jimages, num_bonds, distance_vectors)

            return edge_index_new, -edge_vector_new

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        #edge_index=[2, 10860], y=[101, 1], frac_coords=[1204, 3], atom_types=[1204], lengths=[101, 3], angles=[101, 3], to_jimages=[10860, 3], num_atoms=[101], num_bonds=[101], num_nodes=1204, batch=[1204], ptr=[102]
        node2graph = batch.batch
        #exit()
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        #print(lattices,lattices.device)
        #exit()
        edges_index, edge_vec = self.gen_edges(batch.num_atoms, batch.frac_coords, lattices, node2graph, edge_style='knn')
        #create_model = model_entrypoint('graph_attention_transformer_nonlinear_l2_e3')
        #model = create_model(irreps_in='100x0e',radius=8,num_basis=128,out_channels=1,atomref=None,drop_path=0.0)
        #print(edges_index)
        #exit()
        #model.to(self.device)
        #print(self.device)
        #exit()
        preds = self.trans_model(batch=node2graph,node_atom=batch.atom_types, edge_src=edges_index[1], edge_dst=edges_index[0], edge_attr=edge_vec.norm(dim=1), edge_vec=edge_vec)
        #print(batch.frac_coords)
        #exitg()
        #preds = self.encoder(batch)  # shape (N, 1)

        return preds

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        loss = self.get_loss(preds, batch.y)

        loss = loss.mean()



        self.log_dict(
            {'train_loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )


        if torch.isnan(loss):
            loss = None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        log_dict, loss = self.compute_stats(batch, preds, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        log_dict, loss = self.compute_stats(batch, preds, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, batch, preds, prefix):
        loss = self.get_loss(preds, batch.y)
        loss = self.nanmean(loss)

        self.scaler.match_device(preds)

        if self.task == 'regression':
            
            pcc = self.pearson_correlation(preds.reshape(-1), batch.y.reshape(-1))

            log_dict = {
                f'{prefix}_loss': loss,
                f'{prefix}_pcc': pcc,
            }
        else:
            dis_preds = preds.argmax(dim=-1)
            dis_y = batch.y.reshape(-1)
            acc = torch.mean(dis_preds == dis_y)
            log_dict = {
                f'{prefix}_loss': loss,
                f'{prefix}_acc': acc,
            }        

        return log_dict, loss



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    return model


if __name__ == "__main__":
    main()
