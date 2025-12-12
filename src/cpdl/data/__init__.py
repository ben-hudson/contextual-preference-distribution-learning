import torch

from ..opt_models import BaseGurobiModel
from .grid import *


def generate_instances(
    model: BaseGurobiModel,
    n_instances: int,
    n_samples_per_instance: int,
    n_feats: int,
    noise_scale: float = 1.0,
):
    util_mean_coeffs = torch.rand(n_feats, 1)
    util_scale_coeffs = torch.rand(n_feats, 1)

    n_edges = len(model.edge_list)

    feats = torch.randn(n_instances, n_edges, n_feats)
    util_locs = (feats @ util_mean_coeffs).squeeze(-1)
    util_scales = (torch.nn.functional.softplus(feats @ util_scale_coeffs) * noise_scale).squeeze(-1)

    # X is normally distributed, then Y=exp(X) is log-normally distributed
    noise = torch.randn(n_samples_per_instance, *util_scales.shape)
    utils = torch.exp(util_locs + util_scales * noise)

    sols, _ = model.solve_batch(utils)

    return feats, utils, util_locs, util_scales, sols
