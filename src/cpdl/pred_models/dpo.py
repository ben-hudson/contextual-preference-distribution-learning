import torch

from pyepo.func.perturbed import perturbedOpt
from pyepo.model.opt import optModel

from ..opt_models import BaseGurobiModel
from .base_model import BaseLightningModel
from .utils import get_r2


class DPO(BaseLightningModel):
    def __init__(
        self,
        encoder: torch.nn.Module,
        fwd_opt: optModel,
        decision_policy: BaseGurobiModel,
        n_samples: int = 400,
        noise_scale: float = 0.1,
        loss: str = "mse",
        **kwargs,
    ):
        super().__init__(decision_policy, **kwargs)

        self.encoder = encoder

        self.smooth_fwd_opt = perturbedOpt(fwd_opt, n_samples=n_samples, sigma=noise_scale, processes=1)
        self.noise_scale = noise_scale

        if loss == "mse":
            self.loss_fn = torch.nn.MSELoss()
        # elif loss == "bce":
        #     self.loss_fn = torch.nn.BCELoss()
        else:
            raise ValueError(f"Expected mse or bce, but got loss={loss}.")

    def forward(self, feats: torch.Tensor):
        batch_shape = feats.shape[:-1]
        return self.encoder(feats.flatten(0, -2)).reshape(*batch_shape)

    def training_step(self, batch):
        feats, _, _, _, choices = batch
        choice_prob = choices.mean(dim=1)

        costs_pred = self.forward(feats)
        choice_prob_pred = self.smooth_fwd_opt(costs_pred)

        loss = self.loss_fn(choice_prob_pred, choice_prob)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, log_prefix: str = "val/"):
        feats, costs, cost_locs, _, choices = batch
        costs_pred = self.forward(feats)

        choice_prob = choices.mean(dim=1)
        choice_prob_pred = self.smooth_fwd_opt(costs_pred)
        loss = self.loss_fn(choice_prob_pred, choice_prob)

        cost_locs_flat = cost_locs.cpu().flatten()
        cost_locs_pred_flat = costs_pred.cpu().flatten()
        r2_loc = get_r2(cost_locs_flat, cost_locs_pred_flat)

        metric_dict = {"loss": loss, "r2_mean": r2_loc}
        self.log_dict({log_prefix + k: v for k, v in metric_dict.items()}, prog_bar=True)
        return loss

    def predict_cost_dist(self, feats: torch.Tensor):
        costs_pred = self.forward(feats)
        return torch.distributions.Normal(loc=costs_pred, scale=self.noise_scale)
