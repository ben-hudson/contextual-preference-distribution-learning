import torch

from ..opt_models import ScenarioBasedCVaRMatching
from .base_model import BaseLightningModel


class Oracle(BaseLightningModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # we need a dummy parameter so configure_optimizers doesn't bork
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def training_step(self, batch):
        pass

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, log_prefix="test/")

        feats, costs, cost_locs, cost_scales, _ = batch
        # pick a random feature to be the org costs (e.g. one of the features is travel time)
        org_costs = feats[..., 0]

        cost_dist = torch.distributions.LogNormal(loc=cost_locs, scale=cost_scales)
        if isinstance(self.policy, ScenarioBasedCVaRMatching):
            sols, objs_pred = self.policy.get_scenarios_and_solve_batch(cost_dist, org_costs)
        else:
            sols, objs_pred = self.policy.get_cost_matrix_and_solve_batch(cost_dist, org_costs)

        # realized cost matrices
        cost_matrix = self.policy.get_cost_matrix(costs, org_costs.unsqueeze(1).expand_as(costs))
        # average realized objective value over samples
        objs_real = (cost_matrix * sols.unsqueeze(1)).flatten(start_dim=-2).sum(dim=-1)

        self.log_dict(
            {
                # this is the metric we used when making the plots originally - it throws a warning about broadcasting
                "test/obj_mse": torch.nn.functional.mse_loss(objs_pred, objs_real, reduction="mean"),
                # this is the metric we have in the paper: the expected realized cost minus the expected predicted cost, squared and meaned
                "test/pds": (objs_real.mean(dim=-1, keepdim=True) - objs_pred).pow(2).mean(),
                # I think this is a better way of measuring it, because we get the average squared deviation per realization
                # This metric is the same as the one used making the plots. We need to update the equation in the paper.
                "test/pds_better": (objs_real.swapaxes(0, 1) - objs_pred.swapaxes(0, 1)).pow(2).mean(),
                # these are the post-decision disappointment equivalents - only count when the realized cost is greater than the predicted cost
                "test/pdd": (objs_real.mean(dim=-1, keepdim=True) - objs_pred).clamp(0).pow(2).mean(),
                "test/pdd_better": (objs_real.swapaxes(0, 1) - objs_pred.swapaxes(0, 1)).clamp(0).pow(2).mean(),
            }
        )
