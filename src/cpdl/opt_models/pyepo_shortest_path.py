import gurobipy as gp
import networkx as nx
import numpy as np

from gurobipy import GRB
from pyepo.model.grb.grbmodel import optGrbModel
from typing import Any


class PyEPOShortestPath(optGrbModel):
    def __init__(self, graph: nx.MultiDiGraph, source: Any, sink: Any, is_integer: bool = False):
        # the attributes must match the names in the constructor exactly
        self.graph = graph
        self.source = source
        self.sink = sink
        self.is_integer = is_integer
        super().__init__()

    def _getModel(self):
        model = gp.Model(self.__class__.__name__)

        node_list = list(self.graph.nodes)
        edge_list = list(self.graph.edges)

        if self.is_integer:
            x = model.addMVar(len(edge_list), vtype=GRB.BINARY, name="x")
        else:
            x = model.addMVar(len(edge_list), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

        A = nx.incidence_matrix(self.graph, nodelist=node_list, edgelist=edge_list, oriented=True)
        b = np.zeros(len(node_list))
        b[node_list.index(self.source)] = -1
        b[node_list.index(self.sink)] = 1
        model.addMConstr(A.toarray(), x, "=", b)

        return model, x
