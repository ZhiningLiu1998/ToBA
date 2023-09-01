# %%

import numpy as np
import torch
from args import parse_args
from baselines import *
from data_utils import (
    get_data_stat,
    get_natural_imbalanced_split_data,
    get_step_imbalanced_split_data,
    load_data,
)
from toba import TopoBalanceAugmenter
from trainer import NodeClassificationTrainer
from utils import get_model, get_device, print_centered

MODE_SPACE = ["dummy", "pred", "topo"]


def train(args):
    # get the device for computation
    device = get_device(args.gpu_id)

    # print the arguments for the experiment
    print_centered("Arguments", 40, fillchar="=")
    kwlen = max([len(k) for k in args.__dict__.keys()]) + 1
    for keys, values in args.__dict__.items():
        print(f"{keys:{kwlen}}: {values}")
    print_centered("", 40, fillchar="=")

    # decide the ToBA mode that will be tested
    if args.toba_mode in MODE_SPACE:
        mode_space = [args.toba_mode]
    elif args.toba_mode == "all":
        mode_space = MODE_SPACE
    else:
        raise ValueError(
            f"toba_mode must be one of {MODE_SPACE + ['all']}, got {args.toba_mode}."
        )

    # run the experiment
    for toba_mode in mode_space:
        print_centered(
            f"Dataset [{args.dataset.title()}] - {args.imb_type.title()}IR [{args.imb_ratio}] - ToBA [{toba_mode}]",
            width=80,
            fillchar="=",
            prefix="\n",
        )

        best_results = []
        for i_run in range(1, args.n_runs + 1):
            seed = args.seed + i_run

            # load imbalanced data
            data = load_data(args.dataset, to_device=device, verbose=args.debug)
            if args.imb_type == "step":
                data = get_step_imbalanced_split_data(
                    data, imbratio=args.imb_ratio, random_seed=seed, verbose=args.debug
                )
            elif args.imb_type == "natural":
                data = get_natural_imbalanced_split_data(
                    data, imbratio=args.imb_ratio, random_seed=seed, verbose=args.debug
                )
            else:
                raise ValueError(
                    f"imb_type must be one of ['step', 'natural'], got {args.imb_type}."
                )
            data = get_data_stat(data, store_in_data=True, verbose=args.debug)

            # initialize model
            model = get_model(
                gnn_arch=args.gnn_arch,
                feat_dim=data.n_feat,
                hid_dim=args.hid_dim,
                out_dim=data.n_class,
                n_layer=args.n_layer,
                imb_handling=args.imb_handling,
                device=device,
            )
            # toba augmenter
            augmenter = TopoBalanceAugmenter(mode=toba_mode, random_state=seed)
            # trainer
            trainer = NodeClassificationTrainer(
                model=model,
                data=data,
                device=device,
                augmenter=augmenter,  # ToBA augmentation, to disable, set augmenter=None
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                train_epoch=args.epochs,
                early_stop_patience=args.early_stop,
                eval_freq=1,
                verbose_freq=None,
                enable_tqdm=args.tqdm,
                random_state=seed,
            )
            # train the GNN with ToBA augmentation
            best_model = trainer.train()
            # print best results
            trainer.print_best_results()
            # save best results
            best_results.append(trainer.best_eval_results)

        # print the average performance of the best model
        info = f"Avg Test Performance ({args.n_runs} runs): "
        for metric in trainer.eval_metrics.keys():
            scores = np.array(
                [
                    best_results[i][metric]["test"] * 100
                    for i in range(len(best_results))
                ]
            )
            info += f" | {metric.upper()}: {scores.mean():.2f} ± {scores.std()/(len(scores)**0.5):.2f}"
        print(info)


if __name__ == "__main__":
    # import sys
    # sys.argv = [""]
    args = parse_args()
    train(args)

    # %%
