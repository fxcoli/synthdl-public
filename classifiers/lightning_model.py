import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from pathlib import Path
from torchvision.transforms import v2
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        folder: Path,
        pathology_df: pd.DataFrame,
        size=(256, 256),
        is_train: bool = True,
    ):
        """
        Create an image dataset based on a directory of images

        Arguments:
            folder -- Single directory containing all images
            pathology_df -- DataFrame containing pathology information for
            images, the first column should be the image name and the following
            columns should contain pathology info

        Keyword Arguments:
            size -- _description_ (default: {(256, 256)})
        """
        self.size = size
        self.is_train = is_train

        transform_list = [
            v2.ToImage(),
            v2.Resize(self.size, antialias=True),
            # v2.Grayscale()
        ]

        if self.is_train:
            transform_list.extend(
                [
                    v2.RandomRotation(180),
                    v2.RandomVerticalFlip(),
                    v2.RandomAutocontrast(),
                    v2.RandomResizedCrop(self.size, antialias=True),
                ]
            )

        transform_list.extend(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.449], std=[0.226]),
            ]
        )

        self.transform = v2.Compose(transform_list)
        self.df = pathology_df.copy()
        if "Prompt" in self.df.columns:
            self.df = self.df.drop(columns=["Prompt"])

        # Map filenames to full paths
        if folder is not None:
            if not folder.is_dir():
                raise NotADirectoryError(f"Directory not found: {folder}")
            id_col = self.df.columns[0]
            self.df[id_col] = self.df[id_col].apply(lambda x: folder / x)

        # Infer number of classes (all columns after the first filename column)
        self.num_labels = len(self.df.columns) - 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row.iloc[0]).convert("RGB")

        # Ensure correct tensor shape: [1] for binary, [N] for multi-class
        labels = torch.tensor(row.iloc[1:].tolist(), dtype=torch.float32)
        if self.num_labels == 1:
            labels = labels.view(1)

        return self.transform(image), labels

    def num_classes(self):
        return self.num_labels

    @classmethod
    def merge(cls, dataset1, dataset2, limit=None, ratio=0.5, random_state=42):
        if type(dataset1) is not type(dataset2):
            raise TypeError("Both datasets must be of the same type.")

        if ratio == 0.0:
            return dataset1
        if ratio == 1.0:
            return dataset2

        limit = limit or min(
            int(len(dataset1) / (1 - ratio)), int(len(dataset2) / ratio)
        )
        n1 = int(round(limit * (1 - ratio)))
        n2 = limit - n1

        sample1 = dataset1.df.sample(n1, random_state=random_state)
        sample2 = dataset2.df.sample(n2, random_state=random_state)

        return cls(
            folder=None,
            pathology_df=pd.concat((sample1, sample2), ignore_index=True),
            size=dataset1.size,
            is_train=dataset1.is_train,
        )


class ImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_csv: Path | str,
        test_csv: Path | str,
        batch_size: int = 32,
        img_size: tuple = (256, 256),
        use_pin_memory: bool = False,
    ):
        super().__init__()

        self.train_csv = Path(train_csv)
        self.test_csv = Path(test_csv)

        self.batch_size = batch_size
        self.img_size = img_size
        self.pin_memory = use_pin_memory

        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_df = pd.read_csv(self.train_csv)
            self.train_dataset = ImageDataset(
                folder=self.train_csv.parent,
                pathology_df=train_df,
                size=self.img_size,
                is_train=True,
            )

        if stage in ("fit", "validate") or stage is None:
            test_df = pd.read_csv(self.test_csv)
            self.test_dataset = ImageDataset(
                folder=self.test_csv.parent,
                pathology_df=test_df,
                size=self.img_size,
                is_train=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


class MixedImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_csv_1: Path | str,
        train_csv_2: Path | str,
        test_csv: Path | str,
        mix_ratio: float,
        mix_limit: int,
        batch_size: int = 32,
        img_size: tuple = (256, 256),
        use_pin_memory: bool = False,
        random_state: int = 0,
    ):
        super().__init__()

        self.train_csv_1 = Path(train_csv_1)
        self.train_csv_2 = Path(train_csv_2)
        self.test_csv = Path(test_csv)

        self.batch_size = batch_size
        self.img_size = img_size
        self.mix_ratio = mix_ratio
        self.mix_limit = mix_limit
        self.pin_memory = use_pin_memory
        self.random_state = random_state

        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_df_1 = pd.read_csv(self.train_csv_1)
            dataset_1 = ImageDataset(
                folder=self.train_csv_1.parent,
                pathology_df=train_df_1,
                size=self.img_size,
                is_train=True,
            )
            train_df_2 = pd.read_csv(self.train_csv_2)
            dataset_2 = ImageDataset(
                folder=self.train_csv_2.parent,
                pathology_df=train_df_2,
                size=self.img_size,
                is_train=True,
            )

            self.train_dataset = ImageDataset.merge(
                dataset1=dataset_1,
                dataset2=dataset_2,
                limit=self.mix_limit,
                ratio=self.mix_ratio,
                random_state=self.random_state,
            )

        if stage in ("fit", "validate") or stage is None:
            test_df = pd.read_csv(self.test_csv)
            self.test_dataset = ImageDataset(
                folder=self.test_csv.parent,
                pathology_df=test_df,
                size=self.img_size,
                is_train=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


class BinaryClassifier(L.LightningModule):
    def __init__(
        self,
        classifier: nn.Module,
        loss_fn: torch.nn.Module,
        lr: float,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.classifier = classifier
        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay

        self.train_metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "f1": BinaryF1Score(),
            },
            prefix="train_",
        )
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_epochs = self.trainer.max_epochs
        warmup_epochs = max(1, int(total_epochs * 0.05))

        linear = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=self.lr * 0.01
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer, schedulers=[linear, cosine], milestones=[warmup_epochs]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.classifier(x)
        loss = self.loss_fn(preds, y)

        # Update metrics
        self.train_metrics.update(torch.sigmoid(preds), y.long())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.classifier(x)
        test_loss = self.loss_fn(preds, y)

        self.test_metrics.update(torch.sigmoid(preds), y.long())

        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)


class MultiClassClassifier(L.LightningModule):
    def __init__(
        self,
        classifier: nn.Module,
        loss_fn: torch.nn.Module,
        lr: float,
        num_classes: int,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.classifier = classifier
        self.loss_fn = loss_fn
        self.lr = lr
        self.num_classes = num_classes
        self.weight_decay = weight_decay

        self.train_metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(
                    num_classes=num_classes, average="micro"
                ),
                "precision": MulticlassPrecision(
                    num_classes=num_classes, average="micro"
                ),
                "recall": MulticlassRecall(num_classes=num_classes, average="micro"),
                "f1": MulticlassF1Score(num_classes=num_classes, average="micro"),
            },
            prefix="train_",
        )
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_epochs = self.trainer.max_epochs
        warmup_epochs = max(1, int(total_epochs * 0.05))

        linear = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=self.lr * 0.01,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[linear, cosine],
            milestones=[warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _prepare_targets_for_metrics(self, y: torch.Tensor) -> torch.Tensor:
        """
        Converts 2D one-hot/probability target tensors from ImageDataset
        into 1D integer class indices required by torchmetrics.
        """
        if y.ndim > 1 and y.shape[1] > 1:
            return y.argmax(dim=1)
        return y.long()

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.classifier(x)
        loss = self.loss_fn(preds, y)

        # Convert logits to class probabilities and format target indices
        probs = torch.softmax(preds, dim=1)
        targets = self._prepare_targets_for_metrics(y)

        self.train_metrics.update(probs, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.classifier(x)
        test_loss = self.loss_fn(preds, y)

        probs = torch.softmax(preds, dim=1)
        targets = self._prepare_targets_for_metrics(y)

        self.test_metrics.update(probs, targets)

        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
