import torch

from typing import Callable


class SmoothedFn(torch.nn.Module):
    def __init__(self, fn: Callable, *fn_args, **fn_kwargs):
        super().__init__()
        self.fn = fn
        self.fn_args = fn_args
        self.fn_kwargs = fn_kwargs

    def forward(self, dist: torch.distributions.Distribution, n_samples: int = 1, joint_dim: int = None):
        with torch.no_grad():
            samples = dist.sample((n_samples,))
            fn_outputs = self.fn(samples, *self.fn_args, **self.fn_kwargs)

        log_probs = dist.log_prob(samples)
        if joint_dim is not None:
            log_probs = log_probs.sum(dim=joint_dim)

        smoothed_output = SmoothedFnGradient.apply(fn_outputs, log_probs)
        return smoothed_output


class SmoothedFnGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vals: torch.Tensor, log_probs: torch.Tensor):
        ctx.save_for_backward(vals, log_probs)
        return vals.mean(dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        vals, _ = ctx.saved_tensors

        grad_output_ex = grad_output.unsqueeze(0).expand_as(vals)

        n_samples = vals.size(0)
        log_prob_grad = (vals * grad_output_ex / n_samples).sum(dim=-1)

        return None, log_prob_grad
