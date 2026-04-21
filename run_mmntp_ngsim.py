import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-trajpred")

import torch

import params
from evaluate import test_model_dict
from train import train_model_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--experiment-file", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-itrs", type=int, default=100)
    parser.add_argument("--val-freq", type=int, default=20)
    parser.add_argument("--max-val-itr", type=int, default=2)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--multi-modal-eval", action="store_true")
    return parser.parse_args()


def build_params(args):
    p = params.ParametersHandler("MMnTP.yaml", "ngsim.yaml", "./config")
    p.hyperparams["experiment"]["group"] = "mmntp_ngsim"
    p.hyperparams["experiment"]["debug_mode"] = args.debug
    p.hyperparams["experiment"]["multi_modal_eval"] = args.multi_modal_eval
    p.hyperparams["dataset"]["balanced"] = args.balanced
    p.hyperparams["training"]["batch_size"] = args.batch_size
    p.hyperparams["training"]["num_itrs"] = args.num_itrs
    p.hyperparams["training"]["val_freq"] = args.val_freq
    p.hyperparams["training"]["max_val_itr"] = args.max_val_itr
    p.match_parameters()
    if args.experiment_file:
        p.import_experiment(args.experiment_file)
        p.hyperparams["experiment"]["debug_mode"] = args.debug
        p.hyperparams["experiment"]["multi_modal_eval"] = args.multi_modal_eval
        p.hyperparams["dataset"]["balanced"] = args.balanced
        p.hyperparams["training"]["batch_size"] = args.batch_size
        p.hyperparams["training"]["max_val_itr"] = args.max_val_itr
        p.match_parameters()
    return p


if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.set_sharing_strategy("file_system")

    p = build_params(args)
    if not args.eval_only and not args.experiment_file:
        p.export_experiment()

    if not args.eval_only:
        train_model_dict(p)
    if not args.train_only:
        test_model_dict(p)
