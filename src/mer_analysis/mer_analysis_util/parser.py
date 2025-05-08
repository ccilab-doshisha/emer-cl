import argparse
import json
import os

# ============================== train parser ==============================
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--CUDA_VISIBLE_DEVICES",
    default="0",
    type=str,
    help="CUDA GPU number",
)
parser.add_argument(
    "--epoch", default=10001, type=int, help="epoch num for training"
)
parser.add_argument(
    "--lr", default=1e-5, type=float, help="learning rate for training"
)
parser.add_argument(
    "--batch_size", default=512, type=int, help="batch size for training"
)
parser.add_argument(
    "--embedding_dim",
    default=1024,
    type=int,
    help="embedding dimension will be written. Default : 1024",
)
parser.add_argument(
    "--margin", default=1.0, type=float, help="the margin for rankloss"
)
parser.add_argument(
    "--_lambda",
    default=0.50,
    type=float,
    help="the control parameter lambda (CCALOSS vs RANKLOSS)",
)
parser.add_argument(
    "--dataset",
    default=None,
    type=str,
    required=True,
    help="dataset name (DEAM or PMEmo)",
)

# ============================== train parser ==============================


def save_params(save_dir: str) -> None:
    save_path = os.path.join(save_dir, "params.json")
    with open(save_path, "w") as f:
        json.dump(vars(parser.parse_args()), f)


def load_params(load_dir: str) -> argparse.Namespace:
    load_path = os.path.join(load_dir, "params.json")
    with open(load_path, "r") as f:
        params_json = json.load(f)
    return argparse.Namespace(**params_json)
