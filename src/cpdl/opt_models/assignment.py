import numpy as np
import networkx as nx
import torch

from gurobipy import GRB
from typing import Any, List

from .max_util_flow import ShortestPath
from .base_model import BaseGurobiModel


class OptimalAssignment(BaseGurobiModel):
    def __init__(self, n_agents: int, n_tasks: int):
        super().__init__()

        self._assignments = self._model.addMVar((n_agents, n_tasks), vtype=GRB.BINARY)
        self._model.addConstr(self._assignments.sum(axis=0) == 1)
        self._model.addConstr(self._assignments.sum(axis=1) == 1)

    @property
    def x(self):
        return self._assignments

    @property
    def sense(self):
        return GRB.MINIMIZE

    def set_obj(self, cost_matrix: np.ndarray):
        assert (
            cost_matrix.shape == self.x.shape
        ), f"Expected cost to be a {self.x.shape} matrix but got {cost_matrix.shape}."

        self._model.setObjective(cost_matrix.reshape(-1) @ self.x.reshape(-1), self.sense)


class RiderDriverMatching(OptimalAssignment):
    def __init__(self, graph: nx.MultiDiGraph, riders: List[Any], drivers: List[Any]):
        super().__init__(len(drivers), len(riders))

        self.graph = graph
        self.riders = riders
        self.drivers = drivers

    @property
    def n_drivers(self):
        return len(self.drivers)

    @property
    def n_riders(self):
        return len(self.riders)

    # so this gets a litte complicated...
    # the path is determined according to the generalized driver costs, but
    # we want to minimize some organizational costs in the assignment (e.g. price per mile)
    def get_cost_matrix(self, driver_costs: torch.Tensor, org_costs: torch.Tensor):
        assert driver_costs.shape == org_costs.shape

        batch_shape = driver_costs.shape[:-1]
        cost_matrix = torch.zeros(*batch_shape, self.n_drivers, self.n_riders, dtype=torch.float32)

        for i, driver in enumerate(self.drivers):
            for j, rider in enumerate(self.riders):
                sp = ShortestPath(self.graph, driver, rider)
                sols, _ = sp.solve_batch(driver_costs)
                cost_matrix[..., i, j] = (org_costs * sols).sum(dim=-1)
        return cost_matrix

    def solve_batch(self, driver_costs: torch.Tensor, org_costs: torch.Tensor = None, **kwargs):
        if isinstance(driver_costs, torch.distributions.Distribution):
            driver_costs = driver_costs.mean
        if org_costs is None:
            org_costs = driver_costs

        cost_matrix = self.get_cost_matrix(driver_costs, org_costs)
        return super().solve_batch(cost_matrix, **kwargs)


# Uryasev-Rockafellar method from https://arxiv.org/pdf/1511.00140 section 3.2
class ScenarioBasedCVaRMatching(RiderDriverMatching):
    def __init__(
        self,
        graph: nx.MultiDiGraph,
        riders: List[Any],
        drivers: List[Any],
        n_scenarios: int = 1000,
        alpha: float = 0.95,
    ):
        super().__init__(graph, riders, drivers)

        self._var = self._model.addVar(lb=-GRB.INFINITY, name="VaR")
        self._z = self._model.addMVar(n_scenarios, name="z")
        self._scenario_cost_constr = None
        self.n_scenarios = n_scenarios
        self.alpha = alpha

    @property
    def cost_shape(self):
        return (self.n_scenarios, *self.x.shape)

    def set_obj(self, cost_scenarios: np.ndarray):
        if self._scenario_cost_constr is not None:
            self._model.remove(self._scenario_cost_constr)

        scenario_costs = cost_scenarios.reshape(self.n_scenarios, -1) @ self.x.reshape(-1)
        self._scenario_cost_constr = self._model.addConstr(self._z >= scenario_costs - self._var)
        self._model.setObjective(self._var + (1 / ((1 - self.alpha) * self.n_scenarios)) * self._z.sum(), self.sense)

    def solve(self, cost):
        sol, obj = super().solve(cost)
        # we want to know the expected cost of the solution, rather than the actual objective value (the expected shortfall)
        obj = (cost * sol).sum() / self.n_scenarios
        return sol, obj

    def solve_batch(self, driver_costs: torch.distributions.Distribution, org_costs: torch.Tensor, **kwargs):
        driver_cost_scenarios = driver_costs.sample((self.n_scenarios,))
        org_cost_scenarios = org_costs.expand_as(driver_cost_scenarios)

        # we need the batch dim to come first so we can iterate over it
        driver_cost_scenarios = driver_cost_scenarios.swapaxes(0, 1)
        org_cost_scenarios = org_cost_scenarios.swapaxes(0, 1)

        return super().solve_batch(driver_cost_scenarios, org_cost_scenarios, **kwargs)
