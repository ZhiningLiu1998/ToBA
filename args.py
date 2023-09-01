import argparse

IMB_TYPES = ["step", "natural"]
GNN_ARCHS = ["GCN", "GAT", "SAGE"]
TOBA_MODES = ["dummy", "pred", "topo", "all"]


def parse_args():
    """
    Parses command-line arguments and returns an argparse.Namespace object containing the parsed arguments.

    Returns:
    - args: argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Argument parser for your script.")

    # General
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n_runs", type=int, default=5, help="The number of independent runs"
    )
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode")

    # Dataset
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset Name")
    parser.add_argument(
        "--imb_type", type=str, default="step", choices=IMB_TYPES, help="Imbalance type"
    )
    parser.add_argument("--imb_ratio", type=int, default=10, help="Imbalance Ratio")

    # Architecture
    parser.add_argument("--gnn_arch", type=str, default="GCN", choices=GNN_ARCHS)
    parser.add_argument("--n_layer", type=int, default=3, help="The number of layers")
    parser.add_argument("--hid_dim", type=int, default=256, help="Hidden dimension")

    # Training
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument(
        "--epochs", type=int, default=2000, help="The number of training epochs"
    )
    parser.add_argument(
        "--early_stop", type=int, default=200, help="Early stop patience"
    )
    parser.add_argument(
        "--tqdm", type=bool, default=False, help="Enable tqdm progress bar"
    )

    # Method
    parser.add_argument(
        "--imb_handling",
        type=str,
        default="None",
        help="The imbalance handling strategy to use",
    )

    # ToBA parameters
    parser.add_argument(
        "--toba_mode", type=str, default="all", choices=TOBA_MODES, help="Mode of ToBA"
    )

    # Parse args
    args = parser.parse_args()

    return args
