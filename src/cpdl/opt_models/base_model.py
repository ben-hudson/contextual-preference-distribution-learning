import gurobipy as gp
import numpy as np
import torch

from abc import ABC, abstractmethod
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from gurobipy import GRB
from os import devnull


# silence output to stdout and stderr
# https://stackoverflow.com/a/52442331/2426888
@contextmanager
def hush():
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


class BaseGurobiModel(ABC):
    def __init__(self):
        self._model = gp.Model(self.__class__.__name__)

    # the decision vector
    @property
    @abstractmethod
    def x(self) -> gp.MVar:
        pass

    # the optimization direction
    @property
    @abstractmethod
    def sense(self) -> str:
        pass

    # the shape of the cost vector
    # this is usually the same as the decsion vector...
    # but can be different if we have added auxilliary decsion variables
    @property
    def cost_shape(self) -> tuple:
        return self.x.shape

    def set_obj(self, cost: np.ndarray):
        self._model.setObjective(cost @ self.x, self.sense)

    def solve(self, cost: np.ndarray):
        self.set_obj(cost)
        self._model.update()
        with hush():
            self._model.optimize()

        if self._model.Status != GRB.OPTIMAL:
            raise Exception(f"Solve failed with status {self._model.Status}")

        return self.x.x, self._model.ObjVal

    def solve_batch(self, costs: torch.Tensor, only_sols: bool = False):
        if isinstance(costs, torch.distributions.Distribution):
            costs = costs.mean

        # flatten the batch dims
        costs_np = costs.reshape(-1, *self.cost_shape).cpu().numpy()
        sols_and_objs = map(self.solve, costs_np)
        sols_np, objs_np = zip(*sols_and_objs)

        batch_shape = costs.shape[: -len(self.cost_shape)]
        sols = torch.as_tensor(np.vstack(sols_np)).reshape(*batch_shape, *self.x.shape).float().to(costs.device)
        objs = torch.as_tensor(np.vstack(objs_np)).reshape(*batch_shape, 1).float().to(costs.device)

        if only_sols:
            return sols
        return sols, objs
