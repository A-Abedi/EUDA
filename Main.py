import logging
import argparse
import random

import torch
import numpy as np

from Model import EUDA
from Process.Train import train
from Process.Test import test

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params


def setup(args):

    model = EUDA(args.backbone_type, args.backbone_size, args.bottleneck_size,args.num_classes)
    model.to(args.device)

    for name, param in model.named_parameters():
        if "norm" in name or "bottleneck" in name or "head" in name:
            continue

        param.requires_grad = False

    num_params = count_parameters(model)

    logger.info(f"Backbone Type: {args.backbone_type}, "
                f"Backbone Size: {args.backbone_size}, "
                f"Bottleneck Size: {args.bottleneck_size}")
    logger.info("Training parameters %s", num_params)

    return args, model


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", help="Which downstream task.")
    parser.add_argument("--train_list", help="Path of the training data.")
    parser.add_argument("--test_list", help="Path of the test data.")
    parser.add_argument("--num_classes", default=10, type=int, help="Number of classes in the dataset.")
    parser.add_argument("--backbone_type", choices=["ViT-B_16", "DINOv2"],
                        default="DINOv2", help="Which variant to use.")
    parser.add_argument("--backbone_size", choices=["base", "large", "huge"])
    parser.add_argument("--bottleneck_size", choices=["small", "base", "large", "huge"])

    # TODO: Change pretrained path
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--is_test", default=False, action="store_true",
                        help="If in test mode.")
    parser.add_argument("--source_only", default=False, action="store_true",
                        help="Train without SDAL.")
    parser.add_argument("--dataset_path", type=str, help="Base path of the dataset.", required=True)

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=2000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--gpu_id", default=0, type=int,
                        help="ID of GPU")
    args = parser.parse_args()
    args.logger = logger

    logging.warning(f"Adapting {args.train_list} to {args.test_list}")

    # Setup CUDA, GPU & distributed training
    args.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Process device: %s, gpu id: %s" %
                   (args.device, args.gpu_id))

    args, model = setup(args)

    # Set seed
    set_seed(args)

    if args.is_test:
        test(args, model)
    else:
        train(args, model)

if __name__ == "__main__":
    main()