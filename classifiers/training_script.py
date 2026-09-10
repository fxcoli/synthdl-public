# Imports
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger

from lightning_model import (
    ImageDataModule,
    MixedImageDataModule,
    BinaryClassifier,
    MultiClassClassifier,
)
from vision_models import MODEL_IMAGE_SIZES, build_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image Classifier Training Script")

    # --- Data Arguments ---
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "train_csv", type=Path, help="Path to csv of training images"
    )
    data_group.add_argument("test_csv", type=Path, help="Path to csv of testing images")

    # --- Mixing Arguments ---
    mix_group = parser.add_argument_group("Data Mixing Configuration")
    mix_group.add_argument(
        "--mix", action="store_true", help="Mix with another dataset?"
    )
    mix_group.add_argument(
        "--mix_csv", default=None, type=Path, help="Path to csv of mixing images"
    )
    mix_group.add_argument(
        "--mix_ratio", default=0.0, type=float, help="Ratio of data mixing to achieve"
    )
    mix_group.add_argument(
        "--mix_limit", default=None, help="Ratio of data mixing to achieve"
    )

    # --- Model & Training Arguments ---
    train_group = parser.add_argument_group("Training & Model Configuration")
    train_group.add_argument(
        "model",
        choices=list(MODEL_IMAGE_SIZES.keys()),
        help="Torchvision-supported model to train.",
    )
    train_group.add_argument("epochs", type=int, help="Total number of epochs to train")
    train_group.add_argument(
        "--batch_size", default=16, type=int, help="Images per training batch"
    )
    train_group.add_argument(
        "--learning_rate", default=1e-4, type=float, help="Peak learning rate"
    )
    train_group.add_argument(
        "--weight_decay",
        default=1e-4,
        type=float,
        help="L2 regularization penalty for AdamW",
    )
    train_group.add_argument(
        "--dropout",
        default=0.0,
        type=float,
        help="Dropout probability for the classification head",
    )
    train_group.add_argument(
        "--freeze",
        action="store_true",
        help="Freeze all but the final classification layer",
    )
    train_group.add_argument(
        "--seed", default=0, type=int, help="Global random seed for reproducibility"
    )
    # Dropped saving & loading for now

    wandb_group = parser.add_argument_group("Weights & Biases Configuration")
    wandb_group.add_argument(
        "--name",
        type=str,
        default=None,
        help="Explicit run name for the W&B interface",
    )
    wandb_group.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Space-separated list of tags for W&B filtering (e.g., --tags experiment_1 baseline)",
    )
    wandb_group.add_argument(
        "--group",
        type=str,
        default=None,
        help="Identifier for grouping related runs in the W&B interface",
    )

    return parser.parse_args()


# --- Execution Flow ---
args = parse_args()
seed_everything(args.seed, workers=True)
use_pin_memory = torch.cuda.is_available()
img_size = MODEL_IMAGE_SIZES[args.model]

print("Building dataset...")
if args.mix:
    if not args.mix_csv:
        raise ValueError("--mix requires the --mix_csv argument")

    imgs = MixedImageDataModule(
        train_csv_1=args.train_csv,
        train_csv_2=args.mix_csv,
        test_csv=args.test_csv,
        mix_ratio=args.mix_ratio,
        mix_limit=args.mix_limit,
        batch_size=args.batch_size,
        img_size=(img_size, img_size),
        use_pin_memory=use_pin_memory,
        random_state=args.seed,
    )
    print(f"Mixed datasets at a {args.mix_ratio} ratio")
else:
    imgs = ImageDataModule(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        img_size=(img_size, img_size),
        use_pin_memory=use_pin_memory,
    )

# Explicitly trigger dataset setup to inspect target dimensionality
imgs.setup(stage="fit")
num_classes = imgs.train_dataset.num_classes()
print(f"Detected {num_classes} target label column(s) in training dataset.")

print("Initialising model...")
net = build_classifier(
    model_name=args.model,
    freeze_backbone=args.freeze,
    num_classes=num_classes,
)

if num_classes == 1:
    model = BinaryClassifier(
        classifier=net,
        loss_fn=nn.BCEWithLogitsLoss(),
        lr=args.learning_rate,
    )
else:
    model = MultiClassClassifier(
        classifier=net,
        loss_fn=nn.CrossEntropyLoss(),
        lr=args.learning_rate,
        num_classes=num_classes,
    )

print("Setting up loggers...")
logger = WandbLogger(
    entity="synthdl",
    project="bone_classifiers",
    name=args.name,
    group=args.group,
    tags=args.tags,
    config=args,
)
trainer = Trainer(
    max_epochs=args.epochs,
    logger=logger,
    enable_checkpointing=False,
    deterministic=True,
)

print("Beginning training...")
trainer.fit(model, imgs)