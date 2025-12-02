import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse

from dataset import get_data
from method import AnomalyLVM_Trainer
from tools import write2csv, setup_paths, setup_seed, Logger, log_metrics

setup_seed(111)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up workspace paths
    model_name, image_dir, csv_path, log_path, ckpt_path, tb_logger = setup_paths(args)
    logger = Logger(log_path)
    logger.info(f"Model name: {model_name}")

    # Load model config
    config_path = os.path.join('./model_configs', f'{args.model}.json')
    with open(config_path, 'r') as f:
        model_cfg = json.load(f)

    # Build feature hierarchy
    layers = model_cfg['vision_cfg']['layers']
    step = layers // 4
    feat_list = [step, step * 2, step * 3, step * 4]

    # Initialize AnomalyLVM
    model = AnomalyLVM_Trainer(
        backbone=args.model,
        feat_list=feat_list,
        input_dim=model_cfg['vision_cfg']['width'],
        output_dim=model_cfg['embed_dim'],
        learning_rate=args.lr,
        device=device,
        image_size=args.image_size,
        prompting_depth=args.prompt_depth,
        prompting_length=args.prompt_len,
        prompting_branch=args.prompt_branch,
        prompting_type=args.prompt_type,
        use_hsf=args.use_hsf,
        k_clusters=args.k_clusters
    ).to(device)

    # Load train/test datasets
    train_cls, train_set, train_root = get_data(
        args.train_data, transform=model.preprocess,
        target_transform=model.transform, training=True
    )
    test_cls, test_set, test_root = get_data(
        args.test_data, transform=model.preprocess,
        target_transform=model.transform, training=False
    )

    logger.info(f"Training data: {train_root}")
    logger.info(f"Testing data: {test_root}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    best_f1 = -1e9

    # --------------------- Training Loop --------------------- #
    for epoch in tqdm(range(args.epochs), desc="Training"):
        train_loss = model.train_epoch(train_loader)

        logger.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}")
        tb_logger.add_scalar("loss", train_loss, epoch)

        # Run validation
        metric_dict = model.evaluation(test_loader, test_cls, False, image_dir)
        log_metrics(metric_dict, logger, tb_logger, epoch)

        val_f1 = metric_dict['Average']['f1_px']

        # Save the best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save(ckpt_path + '_best.pth')
            for k in metric_dict.keys():
                write2csv(metric_dict[k], test_cls, k, csv_path)

    logger.info("Training completed.")


def str2bool(x):
    return x.lower() in ["yes", "true", "1", "t"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AnomalyLVM")

    # Dataset
    parser.add_argument("--train_data", nargs="+",
                        default=["mvtec", "colondb"])
    parser.add_argument("--test_data", type=str,
                        default="visa")

    # Model & training config
    parser.add_argument("--model", type=str,
                        default="ViT-L-14-336",
                        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=518)

    # Prompt settings
    parser.add_argument("--prompt_depth", type=int, default=4)
    parser.add_argument("--prompt_len", type=int, default=5)
    parser.add_argument("--prompt_type", type=str, default="SD")
    parser.add_argument("--prompt_branch", type=str, default="VL")
    parser.add_argument("--use_hsf", type=str2bool, default=True)
    parser.add_argument("--k_clusters", type=int, default=20)

    # Workspace
    parser.add_argument("--save_path", type=str, default="./workspaces")

    args = parser.parse_args()

    # Only support batch size = 1
    if args.batch_size != 1:
        raise NotImplementedError("Only batch_size=1 is supported.")

    train(args)
