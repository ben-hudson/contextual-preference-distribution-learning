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
        r2_mean = get_r2(batch.cost_locs, -rewards)

        metric_dict = {"loss": loss, "r2_mean": r2_mean}
        self.log_dict({log_prefix + k: v for k, v in metric_dict.items()}, prog_bar=True)
        return loss
