import argparse
import lightning as L
import pathlib
import random
import torch
import wandb

from cpdl.data import generate_instances, grid_graph
from cpdl.opt_models import ShortestPath, RiderDriverMatching, ScenarioBasedCVaRMatching, PyEPOShortestPath
from cpdl.pred_models import ContextualIO, DPO, REINFORCE, IRL, AIMLE
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from slurm import show_this_job
from torch_geometric.nn import MLP
from torch.utils.data import random_split, DataLoader, TensorDataset


def get_args(project_name):
    parser = argparse.ArgumentParser(project_name)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--decision_policy", type=str, default="ra", choices=["rn", "ra"])
    parser.add_argument("--env_budget", type=int, default=10)
    parser.add_argument("--env_grid_h", type=int, default=5)
    parser.add_argument("--env_grid_w", type=int, default=5)
    parser.add_argument("--env_n_feats", type=int, default=5)
    parser.add_argument("--env_n_items", type=int, default=40)
    parser.add_argument("--env_noise_scale", type=float, default=0.5)
    parser.add_argument("--env", type=str, choices=["knapsack", "sp"], default="sp")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--force_wandb_on", action="store_true")
    parser.add_argument("--loss", type=str, default="mse", choices=["bce", "mse"])
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--lr_sched_patience", type=int, default=20)
    parser.add_argument("--lr_sched_rel_thresh", type=float, default=1e-3)
    parser.add_argument("--lr_start", type=float, default=1e-2)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--model_f_max_iter", type=int, default=1000)
    parser.add_argument("--model_f_tol", type=float, default=1e-6)
    parser.add_argument("--model_noise_scale", type=float, default=0.5)
    parser.add_argument("--model_samples", type=int, default=400)
    parser.add_argument("--model_two_sided_perturbation", type=int, choices=[0, 1], default=0)
    parser.add_argument("--model", type=str, choices=["ours", "dpo", "reinforce", "irl", "imle"], default="ours")
    parser.add_argument("--n_instances", type=int, default=100)
    parser.add_argument("--n_samples_per_instance", type=int, default=1000)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_on_cpu", action="store_true")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = torch.randint(0, 2**16, (1,)).item()

    return args


if __name__ == "__main__":
    project_name = pathlib.Path(__file__).stem
    args = get_args(project_name)

    config = vars(args)

    wandb_mode = "disabled"
    if not args.debug:
        slurm_job_info = show_this_job()
        if slurm_job_info is not None and slurm_job_info["JobName"] != "mila-code":
            wandb_mode = "online"
            config.update(**slurm_job_info)
    if args.force_wandb_on:
        wandb_mode = "online"

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.env == "sp":
        grid = grid_graph(args.env_grid_w, args.env_grid_h, bidirectional=True)
        node_list = list(grid.nodes)
        orig = node_list[0]
        dest = node_list[-1]

        fwd_opt = ShortestPath(grid, orig, dest)

        # necessary for DPO model
        pyepo_fwd_opt = PyEPOShortestPath(grid, orig, dest)

        riders = random.choices(node_list, k=3)
        driver_list = list(set(node_list) - set(riders))
        drivers = random.choices(driver_list, k=3)
        if args.decision_policy == "rn":
            decision_policy = RiderDriverMatching(grid, riders, drivers)
        elif args.decision_policy == "ra":
            decision_policy = ScenarioBasedCVaRMatching(grid, riders, drivers, n_scenarios=1000, alpha=0.95)
        else:
            raise ValueError

    # elif args.env == "knapsack":
    #     fwd_opt = IntegerKnapsack(args.env_n_items, args.env_budget)
    #     decision_policy = IntegerKnapsack(args.env_n_items, args.env_budget)

    else:
        raise ValueError(f"Unknown argument env={args.env}.")

    feats, perceived_costs, cost_locs, cost_scales, sols = generate_instances(
        fwd_opt,
        args.n_instances + args.n_test,
        args.n_samples_per_instance,
        args.env_n_feats,
        noise_scale=args.env_noise_scale,
    )
    perceived_costs = perceived_costs.swapaxes(0, 1)
    sols = sols.swapaxes(0, 1)

    n_val = max(1, int(args.n_instances * 0.2))
    n_train = args.n_instances - n_val
    train_data, val_data, test_data = random_split(
        TensorDataset(feats, perceived_costs, cost_locs, cost_scales, sols), [n_train, n_val, args.n_test]
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, num_workers=args.n_workers, persistent_workers=True, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, num_workers=args.n_workers, persistent_workers=True, shuffle=False
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, num_workers=args.n_workers, persistent_workers=True, shuffle=False
    )

    if args.model == "ours":
        encoder = MLP([args.env_n_feats, 32, 32, 2])
        model = ContextualIO(
            encoder,
            fwd_opt,
            decision_policy,
            n_samples=args.model_samples,
            loss=args.loss,
            lr_start=args.lr_start,
            lr_stop=args.lr_min,
            lr_patience=args.lr_sched_patience,
            lr_rel_tol=args.lr_sched_rel_thresh,
        )
    elif args.model == "dpo":
        encoder = MLP([args.env_n_feats, 32, 32, 1])
        model = DPO(
            encoder,
            pyepo_fwd_opt,
            decision_policy,
            n_samples=args.model_samples,
            noise_scale=args.model_noise_scale,
            lr_start=args.lr_start,
            lr_stop=args.lr_min,
            lr_patience=args.lr_sched_patience,
            lr_rel_tol=args.lr_sched_rel_thresh,
        )
    elif args.model == "reinforce":
        encoder = MLP([args.env_n_feats, 32, 32, 2])
        model = REINFORCE(
            encoder,
            fwd_opt,
            decision_policy,
            n_samples=args.model_samples,
            loss=args.loss,
            lr_start=args.lr_start,
            lr_stop=args.lr_min,
            lr_patience=args.lr_sched_patience,
            lr_rel_tol=args.lr_sched_rel_thresh,
        )
    elif args.model == "irl":
        encoder = MLP([args.env_n_feats, 32, 32, 1])
        model = IRL(
            encoder,
            grid,
            orig,
            dest,
            decision_policy,
            f_max_iter=args.model_f_max_iter,
            f_solver="fixed_point_iter",
            f_tol=args.model_f_tol,
            lr_patience=args.lr_sched_patience,
            lr_rel_tol=args.lr_sched_rel_thresh,
            lr_start=args.lr_start,
            lr_stop=args.lr_min,
        )
    elif args.model == "imle":
        encoder = MLP([args.env_n_feats, 32, 32, 1])
        model = AIMLE(
            encoder,
            pyepo_fwd_opt,
            decision_policy,
            n_samples=args.model_samples,
            noise_scale=args.model_noise_scale,
            both_sides=bool(args.model_two_sided_perturbation),
            lr_start=args.lr_start,
            lr_stop=args.lr_min,
            lr_patience=args.lr_sched_patience,
            lr_rel_tol=args.lr_sched_rel_thresh,
        )
    else:
        raise ValueError

    wandb_logger = WandbLogger(
        experiment=wandb.init(project=project_name, group=args.exp_name, mode=wandb_mode, config=config)
    )

    trainer = L.Trainer(
        callbacks=[LearningRateMonitor(logging_interval="epoch")],
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        fast_dev_run=args.debug,
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
