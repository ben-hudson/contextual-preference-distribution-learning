import torch

from ..opt_models import BaseGurobiModel
from .base_model import BaseLightningModel
from .utils import get_r2


class REINFORCE(BaseLightningModel):
    def __init__(
        self,
        encoder: torch.nn.Module,
        fwd_opt: BaseGurobiModel,
        decision_policy: BaseGurobiModel,
        n_samples: int = 400,
        loss: str = "mse",
        **kwargs,
    ):
        super().__init__(decision_policy, **kwargs)

        self.encoder = encoder

        self.fwd_opt = fwd_opt
        self.n_samples = n_samples

        if loss == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="none")
        # elif loss == "bce":
        #     self.loss_fn = torch.nn.BCELoss(reduction="none")
        else:
            raise ValueError(f"Expected mse or bce, but got loss={loss}.")

    def forward(self, feats: torch.Tensor):
        batch_shape = feats.shape[:-1]
        dist_params = self.encoder(feats.flatten(0, -2)).reshape(*batch_shape, -1)
        loc, scale = dist_params.chunk(dist_params.size(-1), dim=-1)

        loc = loc.squeeze(-1)
        scale = torch.nn.functional.softplus(scale).squeeze(-1)
        cost_dists = torch.distributions.LogNormal(loc=loc, scale=scale)
        cost_dist = torch.distributions.Independent(cost_dists, 1)
        return cost_dist

    def training_step(self, batch):
        feats, _, _, _, choices = batch
        choice_prob = choices.mean(dim=1)

        cost_dist = self.forward(feats)

        with torch.no_grad():
            samples = cost_dist.sample((self.n_samples,))
            sampled_choices, _ = self.fwd_opt.solve_batch(samples)
            choice_targets = choice_prob.expand_as(sampled_choices)
            rewards = -self.loss_fn(sampled_choices, choice_targets).sum(dim=-1)
            # normalize
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)

        log_probs = cost_dist.log_prob(samples)
        loss = (-rewards * log_probs).sum()

        self.log_dict({"train/loss": loss}, prog_bar=True)
        return loss

    def predict_cost_dist(self, feats):
        return self.forward(feats)

    def get_prob(self, cost_dist: torch.distributions.Distribution):
        samples = cost_dist.sample((self.n_samples,))
        sampled_choices, _ = self.fwd_opt.solve_batch(samples)
        return sampled_choices.mean(dim=0)

    def validation_step(self, batch, batch_idx, log_prefix: str = "val/"):
        feats, costs, cost_locs, cost_scales, choices = batch
        choice_prob = choices.mean(dim=1)

        cost_dist_pred = self.forward(feats)
        choice_prob_pred = self.get_prob(cost_dist_pred)

        loss = self.loss_fn(choice_prob_pred, choice_prob).mean()

        cost_means_flat = cost_locs.cpu().flatten()
        cost_scales_flat = cost_scales.cpu().flatten()

        cost_means_pred_flat = cost_dist_pred.base_dist.loc.cpu().flatten()
        cost_scales_pred_flat = cost_dist_pred.base_dist.scale.cpu().flatten()

        r2_mean = get_r2(cost_means_flat, cost_means_pred_flat)
        r2_scale = get_r2(cost_scales_flat, cost_scales_pred_flat)

        metric_dict = {"loss": loss, "r2_mean": r2_mean, "r2_scale": r2_scale}
        self.log_dict({log_prefix + k: v for k, v in metric_dict.items()}, prog_bar=True)
        return loss
