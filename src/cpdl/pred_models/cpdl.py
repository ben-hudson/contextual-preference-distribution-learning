import torch

from ..layers.smoothing import SmoothedFn
from ..opt_models import BaseGurobiModel
from .base_model import BaseLightningModel
from .utils import get_r2


class ContextualIO(BaseLightningModel):
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

        self.smooth_fwd_opt = SmoothedFn(fwd_opt.solve_batch, only_sols=True)
        self.n_samples = n_samples

        if loss == "mse":
            self.loss_fn = torch.nn.MSELoss()
        # elif loss == "bce":
        #     self.loss_fn = torch.nn.BCELoss()
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

    def predict_cost_dist(self, feats):
        return self.forward(feats)

    def get_prob(self, feats: torch.Tensor):
        cost_dist = self.forward(feats)
        prob = self.smooth_fwd_opt(cost_dist, n_samples=self.n_samples).clamp(0, 1)
        return prob, cost_dist

    def training_step(self, batch):
        feats, _, _, _, choices = batch
        choice_prob = choices.mean(dim=1)

        choice_prob_pred, _ = self.get_prob(feats)

        loss = self.loss_fn(choice_prob_pred, choice_prob)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, log_prefix: str = "val/"):
        feats, _, cost_locs, cost_scales, choices = batch
        choice_prob = choices.mean(dim=1)

        choice_prob_pred, cost_dist_pred = self.get_prob(feats)

        loss = self.loss_fn(choice_prob_pred, choice_prob)

        cost_means_flat = cost_locs.cpu().flatten()
        cost_scales_flat = cost_scales.cpu().flatten()

        cost_means_pred_flat = cost_dist_pred.base_dist.loc.cpu().flatten()
        cost_scales_pred_flat = cost_dist_pred.base_dist.scale.cpu().flatten()

        r2_mean = get_r2(cost_means_flat, cost_means_pred_flat)
        r2_scale = get_r2(cost_scales_flat, cost_scales_pred_flat)

        # store these as attributes so we can access them in on_validation_epoch_end
        # self.cost_means_list.append(cost_means_flat)
        # self.cost_means_pred_list.append(cost_means_pred_flat)
        # self.cost_scales_list.append(cost_scales_flat)
        # self.cost_scales_pred_list.append(cost_scales_pred_flat)
        metric_dict = {"loss": loss, "r2_mean": r2_mean, "r2_scale": r2_scale}
        self.log_dict({log_prefix + k: v for k, v in metric_dict.items()}, prog_bar=True)
        return loss

    # def on_validation_epoch_start(self):
    #     self.cost_means_list = []
    #     self.cost_means_pred_list = []
    #     self.cost_scales_list = []
    #     self.cost_scales_pred_list = []

    # def on_validation_epoch_end(self):
    #     fig_mean = corr_plot(
    #         torch.cat(self.cost_means_list),
    #         torch.cat(self.cost_means_pred_list),
    #         z_label="$\\theta$",
    #         z_pred_label="$\\theta$ - predicted",
    #     )
    #     fig_scale = corr_plot(
    #         torch.cat(self.cost_scales_list),
    #         torch.cat(self.cost_scales_pred_list),
    #         z_label="$\\theta$",
    #         z_pred_label="$\\theta$ - predicted",
    #     )
    #     self.logger.log_image(
    #         key=f"val/param_recovery",
    #         images=[fig_mean, fig_scale],
    #         caption=["loc", "scale"],
    #         step=self.global_step,
    #     )
    #     plt.close(fig_mean)
    #     plt.close(fig_scale)
