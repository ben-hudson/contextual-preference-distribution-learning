import lightning as L
import torch

from ..opt_models import BaseGurobiModel


# this holds some standard stuff that...
# a) we want to make sure stays the same across models
# b) I don't want to type each time
class BaseLightningModel(L.LightningModule):
    def __init__(
        self,
        decision_policy: BaseGurobiModel,
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
        pass

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, log_prefix="test/")

        feats, costs, _, _, _ = batch
        # pick a random feature to be the org costs (e.g. one of the features is travel time)
        org_costs = feats[..., 0]

        cost_dist_pred = self.predict_cost_dist(feats)
        sols, objs_pred = self.policy.solve_batch(cost_dist_pred, org_costs)

        # realized cost matrices
        cost_matrix = self.policy.get_cost_matrix(costs, org_costs.unsqueeze(1).expand_as(costs))
        # average realized objective value over samples
        objs_real = (cost_matrix * sols.unsqueeze(1)).flatten(start_dim=1).sum(dim=1, keepdim=True) / costs.size(1)

        self.log_dict(
            {
                "test/obj_err": (objs_pred - objs_real).mean(),
                "test/obj_err_rel": ((objs_pred - objs_real) / objs_real).mean(),
                "test/obj_mse": torch.nn.functional.mse_loss(objs_pred, objs_real, reduction="mean"),
                "test/obj_abs_err": torch.nn.functional.l1_loss(objs_pred, objs_real, reduction="mean"),
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
