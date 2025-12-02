import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import argparse
import torch
import cv2
import json
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

from dataset import get_data, dataset_dict
from tools import write2csv, setup_seed, Logger
from method import AnomalyLVM   

setup_seed(111)


def test(args):

    assert os.path.isfile(args.ckt_path), f"Invalid model path: {args.ckt_path}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = Logger("log_test.txt")

    # ------- Print args -------
    for k, v in sorted(vars(args).items()):
        logger.info(f"{k} = {v}")

    # ------- Load AnomalyLVM config -------
    config_path = os.path.join("./model_configs", f"{args.model}.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # ------- Build model -------
    model = AnomalyLVM(
        vision_backbone=args.model,
        vision_dim=cfg["vision_dim"],
        text_dim=cfg["text_dim"],
        decoder_dim=cfg["decoder_dim"],
        device=device,
    ).to(device)

    model.load(args.ckt_path)

    # =====================================================================
    #   TEST ON DATASET
    # =====================================================================
    if args.testing_mode == "dataset":
        assert args.testing_data in dataset_dict.keys(), \
            f"Unsupported dataset {args.testing_data}"

        save_root = args.save_path
        os.makedirs(save_root, exist_ok=True)

        csv_path = os.path.join(save_root, f"{args.testing_data}.csv")
        image_dir = os.path.join(save_root, f"{args.testing_data}_vis")
        os.makedirs(image_dir, exist_ok=True)

        cls_names, dataset, data_root = get_data(
            dataset_type_list=args.testing_data,
            transform=model.preprocess,
            target_transform=model.target_transform,
            training=False,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False
        )

        metric_dict = model.evaluate_dataset(
            dataloader, cls_names, save_fig=args.save_vis, save_dir=image_dir
        )

        # Print metrics
        for name, result in metric_dict.items():
            logger.info(
                f"{name:>15}\t I-AUROC:{result['auroc_im']:.2f}"
                f"\tP-AUROC:{result['auroc_px']:.2f}"
            )

        # Save CSV
        for k in metric_dict.keys():
            write2csv(metric_dict[k], cls_names, k, csv_path)

    # =====================================================================
    #   TEST SINGLE IMAGE
    # =====================================================================
    elif args.testing_mode == "image":
        assert os.path.isfile(args.image_path), f"Invalid image path: {args.image_path}"

        ori = cv2.resize(cv2.imread(args.image_path), (args.image_size, args.image_size))
        pil = Image.open(args.image_path).convert("RGB")

        img = model.preprocess(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            anomaly_map, anomaly_score = model.forward_image(
                img, class_name=args.class_name
            )

        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        anomaly_score = anomaly_score.item()

        # Smooth and Normalize
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        anomaly_map = (anomaly_map * 255).astype(np.uint8)

        heat = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        vis = cv2.addWeighted(heat, 0.5, ori, 0.5, 0)

        result = cv2.hconcat([ori, vis])

        save_path = os.path.join(args.save_path, args.save_name)
        cv2.imwrite(save_path, result)

        print(f"[Saved] {save_path}, anomaly score = {anomaly_score:.3f}")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AnomalyLVM-Test", add_help=True)

    # ---------------- Args ----------------
    parser.add_argument("--ckt_path", type=str, required=True)

    parser.add_argument("--testing_mode", type=str, default="dataset",
                        choices=["dataset", "image"])

    # dataset test
    parser.add_argument("--testing_data", type=str, default="mvtec")

    # single image test
    parser.add_argument("--image_path", type=str, default="asset/img.png")
    parser.add_argument("--class_name", type=str, default="object")
    parser.add_argument("--save_name", type=str, default="result.png")

    parser.add_argument("--model", type=str, default="ViT-L-14-336")
    parser.add_argument("--image_size", type=int, default=518)

    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--save_vis", type=str2bool, default=True)

    args = parser.parse_args()

    test(args)
