import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


def get_r2(z_true: torch.Tensor, z_pred: torch.Tensor):
    z_true = z_true.reshape(-1, 1)
    z_pred = z_pred.reshape(-1, 1)
    # we fit a linear model to predict true latents form the model latents
    linear_model = LinearRegression().fit(z_pred, z_true)
    # and then evaluate how we did with our linear model
    r2 = linear_model.score(z_pred, z_true)
    return r2


def corr_plot(
    z: torch.Tensor, z_pred: torch.Tensor, title: str = "z vs. z_pred", z_label: str = "z", z_pred_label: str = "z_pred"
):
    fig, ax = plt.subplots(1, 1)
    sns.histplot(x=z, y=z_pred, bins=24, fill=True, legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(z_label)
    ax.set_ylabel(z_pred_label)
    return fig
