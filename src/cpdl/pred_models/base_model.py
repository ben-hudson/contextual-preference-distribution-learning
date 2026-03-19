import lightning as L
import torch

from ..opt_models import ScenarioBasedCVaRMatching


# this holds some standard stuff that...
# a) we want to make sure stays the same across models
# b) I don't want to type each time
class BaseLightningModel(L.LightningModule):
    def __init__(
        self,
        decision_policy: ScenarioBasedCVaRMatching,
        lr_start: float = 5e-3,
        lr_stop: float = 1e-4,
        lr_rel_tol: float = 1e-3,
        lr_patience: int = 20,
    ):
        super().__init__()

        self.policy = decision_policy

        self.lr_start = lr_start
        self.lr_stop = lr_stop
        self.lr_rel_tol = lr_rel_tol
        self.lr_patience = lr_patience

    def predict_cost_dist(self, feats: torch.Tensor) -> torch.distributions.Distribution:
        raise NotImplementedError("test step require a function that predicts cost distributions from features")

    def validation_step(self, batch, batch_idx, log_prefix: str = "val/"):
        self.log(log_prefix + "loss", 0)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, log_prefix="test/")

        feats, costs, cost_locs, cost_scales, _ = batch
        # pick a random feature to be the org costs (e.g. one of the features is travel time)
        org_costs = feats[..., 0]

        # cost_dist_pred = self.predict_cost_dist(feats)
        cost_matrix = self.get_cost_matrix(
            feats, self.policy.drivers, self.policy.riders, org_costs, self.policy.n_scenarios
        )
        sols, objs_pred = self.policy.solve_batch(cost_matrix)

        # realized cost matrices
        cost_matrix = self.policy.get_cost_matrix(costs, org_costs.unsqueeze(1).expand_as(costs))
        # average realized objective value over samples
        objs_real = (cost_matrix * sols.unsqueeze(1)).flatten(start_dim=-2).sum(dim=-1)

        self.log_dict(
            {
                "test/pds": (objs_real.mean(dim=-1, keepdim=True) - objs_pred).pow(2).mean(),
                "test/pdd": (objs_real.mean(dim=-1, keepdim=True) - objs_pred).clamp(0).pow(2).mean(),
            }
        )

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr_start)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, threshold=self.lr_rel_tol, threshold_mode="rel", patience=self.lr_patience, min_lr=self.lr_stop
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "name": "lr",
                "scheduler": sched,
                "monitor": "val/loss",
                "frequency": 1,
            },
        }
