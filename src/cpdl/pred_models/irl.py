import torch
import torch_geometric.utils
import networkx as nx

from torch_geometric.data import Data, Batch
from route_choice import MarkovRouteChoice
from typing import Any

from ..opt_models import BaseGurobiModel
from .base_model import BaseLightningModel
from .utils import get_r2


class IRL(BaseLightningModel):
    def __init__(
        self,
        encoder: torch.nn.Module,
        graph: nx.MultiDiGraph,
        orig: Any,
        dest: Any,
        decision_policy: BaseGurobiModel,
        f_tol=1e-6,
        f_solver="fixed_point_iter",
        f_max_iter=1000,
        **kwargs,
    ):
        super().__init__(decision_policy, **kwargs)

        for n in graph.nodes:
            graph.nodes[n]["is_orig"] = n == orig
            graph.nodes[n]["is_dest"] = n == dest
        self.node_list = list(graph.nodes)
        self.graph = torch_geometric.utils.from_networkx(graph)

        self.choice_model = MarkovRouteChoice(encoder, node_dim=-1)
        self.f_tol = f_tol
        self.f_solver = f_solver
        self.f_max_iter = f_max_iter

    def to_torch_geometric_batch(self, feats, costs, cost_locs, cost_scales, sols):
        batch_size = feats.size(0)
        costs = costs.swapaxes(-1, -2)
        sols = sols.swapaxes(-1, -2)

        data_list = []
        for i in range(batch_size):
            sample = Data(
                edge_index=self.graph.edge_index.to(feats.device),
                feats=feats[i],
                costs=costs[i].to(feats.device),
                cost_locs=cost_locs[i].to(feats.device),
                cost_scales=cost_scales[i].to(feats.device),
                sols=sols[i].to(feats.device),
                is_orig=self.graph.is_orig.to(feats.device),
                is_dest=self.graph.is_dest.to(feats.device),
                num_nodes=self.graph.num_nodes,
            )
            data_list.append(sample)
        return Batch.from_data_list(data_list)

    def training_step(self, batch):
        batch = self.to_torch_geometric_batch(*batch)
        rewards, values, action_prob = self.choice_model(
            batch.edge_index,
            batch.feats,
            batch.is_dest,
            f_solver=self.f_solver,
            f_tol=self.f_tol,
            f_max_iter=self.f_max_iter,
        )
        node_flows, edge_flows = self.choice_model.get_flows(
            batch.edge_index,
            action_prob,
            batch.is_orig,
            f_solver=self.f_solver,
            f_tol=self.f_tol,
            f_max_iter=self.f_max_iter,
        )

        loss = torch.nn.functional.mse_loss(edge_flows, batch.sols.mean(dim=1))
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, log_prefix: str = "val/"):
        batch = self.to_torch_geometric_batch(*batch)
        rewards, values, action_prob = self.choice_model(
            batch.edge_index,
            batch.feats,
            batch.is_dest,
            f_solver=self.f_solver,
            f_tol=self.f_tol,
            f_max_iter=self.f_max_iter,
        )
        node_flows, edge_flows = self.choice_model.get_flows(
            batch.edge_index,
            action_prob,
            batch.is_orig,
            f_solver=self.f_solver,
            f_tol=self.f_tol,
            f_max_iter=self.f_max_iter,
        )

        loss = torch.nn.functional.mse_loss(edge_flows, batch.sols.mean(dim=1))
        r2_mean = get_r2(batch.cost_locs.cpu(), -rewards.cpu())
        metric_dict = {"loss": loss, "r2_loc": r2_mean}
        self.log_dict({log_prefix + k: v for k, v in metric_dict.items()}, prog_bar=True)
        return loss

    def sample_path(self, edge_index, action_probs, orig_idx, dest_idx):
        sol = torch.zeros(self.graph.num_edges)

        k = orig_idx
        iters = 0
        while k != dest_idx and iters < 1000:
            action_probs_masked = (edge_index[0] == k) * action_probs
            action = torch.multinomial(action_probs_masked, 1).squeeze()
            sol[action] += 1.0
            k = edge_index[1, action]
            iters += 1

        return sol

    def get_cost_matrix(self, feats, origs, dests, org_costs, n_samples):
        batch_size = feats.size(0)
        edge_index = self.graph.edge_index.to(feats.device)

        org_costs = org_costs.cpu()
        cost_matrix = torch.zeros(batch_size, n_samples, len(origs), len(dests), dtype=torch.float32)

        for l, dest in enumerate(dests):
            dest_idx = self.node_list.index(dest)

            is_dest = torch.zeros(len(self.node_list), device=feats.device)
            is_dest[dest_idx] = 1.0

            data_list = []
            for i in range(batch_size):
                sample = Data(
                    edge_index=edge_index,
                    feats=feats[i],
                    is_dest=is_dest,
                    num_nodes=self.graph.num_nodes,
                )
                data_list.append(sample)
            batch = Batch.from_data_list(data_list)

            rewards, values, action_prob = self.choice_model(
                batch.edge_index,
                batch.feats,
                batch.is_dest,
                f_solver=self.f_solver,
                f_tol=self.f_tol,
                f_max_iter=self.f_max_iter,
            )
            node_batch = batch.batch.to(feats.device)
            edge_batch = node_batch[batch.edge_index[0]]
            action_prob_dense, _ = torch_geometric.utils.to_dense_batch(action_prob, edge_batch)
            action_prob_dense = action_prob_dense.cpu()

            for i in range(batch_size):
                for k, orig in enumerate(origs):
                    orig_idx = self.node_list.index(orig)
                    for j in range(n_samples):
                        path = self.sample_path(self.graph.edge_index, action_prob_dense[i], orig_idx, dest_idx)
                        cost_matrix[i, j, k, l] = org_costs[i] @ path

        return cost_matrix
