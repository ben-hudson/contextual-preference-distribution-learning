import networkx as nx
import numpy as np
import torch

from gurobipy import GRB, nlfunc
from typing import Any

from .base_model import BaseGurobiModel


class ShortestPath(BaseGurobiModel):
    def __init__(self, graph: nx.MultiDiGraph, source: Any, sink: Any, is_integer: bool = False):
        super().__init__()

        self.node_list = list(graph.nodes)
        self.edge_list = list(graph.edges)

        if is_integer:
            self._flows = self._model.addMVar(len(self.edge_list), vtype=GRB.BINARY, name="x")
        else:
            self._flows = self._model.addMVar(len(self.edge_list), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

        self.A = nx.incidence_matrix(graph, nodelist=self.node_list, edgelist=self.edge_list, oriented=True)
        self.b = np.zeros(len(self.node_list))
        self.b[self.node_list.index(source)] = -1
        self.b[self.node_list.index(sink)] = 1
        self._model.addMConstr(self.A.toarray(), self._flows, "=", self.b)

    @property
    def x(self):
        return self._flows

    @property
    def sense(self):
        return GRB.MINIMIZE


class RegularizedShortestPath(ShortestPath):
    def __init__(
        self,
        graph: nx.MultiDiGraph,
        source: Any,
        sink: Any,
        edge_lengths: torch.Tensor = None,
        regularizer: str = "entropy",
    ):
        super().__init__(graph, source, sink)

        if edge_lengths is None:
            edge_lengths = np.zeros(len(graph.edges))
        else:
            edge_lengths = edge_lengths.numpy()

        if regularizer == "entropy":
            # regularizer from "A perturbed utility route choice model"
            reg = edge_lengths @ ((1 + self._flows) * nlfunc.log(1 + self._flows) - self._flows)
        elif regularizer == "square" or regularizer == "l2":
            # another option from "A perturbed utility route choice model"
            # also seen in "Learning with combinatorial optimization layers: a probabilistic approach"
            reg = edge_lengths @ nlfunc.square(self._flows)
        else:
            raise ValueError(f"Unknown value for regularizer: {regularizer}")

        self._reg = self._model.addMVar(reg.shape, vtype=GRB.CONTINUOUS, name="y")
        self._model.addConstr(self._reg == reg)

    def set_obj(self, cost: np.ndarray):
        self._model.setObjective((cost @ self.x) - self._reg, self.sense)
