import torch

from pyepo.func.perturbed import adaptiveImplicitMLE, sumGammaDistribution
from pyepo.model.opt import optModel

from ..opt_models import BaseGurobiModel
from .base_model import BaseLightningModel
from .utils import get_r2


class SumOfGammas(torch.distributions.Distribution):
    def __init__(self, pyepo_dist: sumGammaDistribution, loc: torch.Tensor, scale: torch.Tensor, **kwargs):
        super().__init__(validate_args=False, **kwargs)

        self.loc = loc
        self.scale = scale
        self.unit_dist = pyepo_dist

    def sample(self, sample_shape=...):
        samples_np = self.unit_dist.sample((*sample_shape, *self.loc.shape))
        samples = torch.as_tensor(samples_np).float().to(self.loc.device)
        return self.loc + self.scale * samples


class AIMLE(BaseLightningModel):
    def __init__(
        self,
        encoder: torch.nn.Module,
        fwd_opt: optModel,
        decision_policy: BaseGurobiModel,
        n_samples: int = 400,
        noise_scale: float = 0.1,
        lambd: float = 10,
        both_sides: bool = False,
        loss: str = "mse",
        **kwargs,
    ):
        super().__init__(decision_policy, **kwargs)

        self.encoder = encoder

        # self.smooth_fwd_opt = implicitMLE(
        #     fwd_opt, n_samples=n_samples, sigma=noise_scale, lambd=lambd, two_sides=two_sides
        # )
        self.smooth_fwd_opt = adaptiveImplicitMLE(fwd_opt, n_samples=n_samples, sigma=noise_scale, two_sides=both_sides)

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
        feats, _, cost_locs, _, choices = batch
        costs_pred = self.forward(feats)

        choice_prob = choices.mean(dim=1)
        choice_prob_pred = self.smooth_fwd_opt(costs_pred)
        loss = self.loss_fn(choice_prob_pred, choice_prob)

        cost_means_flat = cost_locs.cpu().flatten()
        cost_means_pred_flat = costs_pred.cpu().flatten()
        r2_mean = get_r2(cost_means_flat, cost_means_pred_flat)

        metric_dict = {"loss": loss, "r2_mean": r2_mean}
        self.log_dict({log_prefix + k: v for k, v in metric_dict.items()}, prog_bar=True)
        return loss

    def predict_cost_dist(self, feats: torch.Tensor):
        costs_pred = self.forward(feats)
        return SumOfGammas(self.smooth_fwd_opt.distribution, loc=costs_pred, scale=self.smooth_fwd_opt.sigma)
