import sys

sys.path.append("../")

import json
import logging
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import OmegaConf

from classification import utils_global
from classification.dataset import MsgPackIterableDatasetMultiTargetWithDynLabels
from classification.optimizers import get_optimizer_parameters
from classification.s2_utils import Hierarchy, Partitioning

import yaml

# ============================================
#                  Changelog
# ============================================
# Switch from tensorboard to wandb logging
# Update tfm according to image_dim
# Add finetune_lr_multiplier
# ============================================
#                  
# ============================================

class MultiPartitioningBase(pl.LightningModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams.update(hparams)
        self.partitionings, self.hierarchy = self.init_partitionings()
        self.image_dim = self.build_model()

        if self.hparams.get("loss_weight", ""):
            loss_weights = json.load(open(hparams["loss_weight"]))
            self.losses = [torch.nn.CrossEntropyLoss(weight=torch.tensor(loss_weight)) for loss_weight in loss_weights]
        else:
            self.losses = [torch.nn.CrossEntropyLoss() for _ in range(len(self.partitionings))]

    def init_partitionings(self):

        partitionings = []
        for shortname, path in zip(
            self.hparams.partitionings["shortnames"],
            self.hparams.partitionings["files"],
        ):
            partitionings.append(Partitioning(Path(path), shortname, skiprows=self.hparams.get("skiprows", 2)))

        if len(self.hparams.partitionings["files"]) == 1:
            return partitionings, None

        return partitionings, Hierarchy(partitionings)

    def build_model(self):
        logging.info("Build model")
        model, nfeatures, image_dim = utils_global.build_base_model(self.hparams.arch)

        classifier = torch.nn.ModuleList(
            [
                torch.nn.Linear(nfeatures, len(self.partitionings[i]))
                for i in range(len(self.partitionings))
            ]
        )

        if self.hparams.weights:
            logging.info("Load weights from pre-trained model")
            model, classifier = utils_global.load_weights_if_available(
                model, classifier, self.hparams.weights
            )

        self.model = model
        self.classifier = classifier
        return image_dim

    def forward(self, x):
        fv = self.model(x)
        yhats = [self.classifier[i](fv) for i in range(len(self.partitionings))]
        return yhats

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        images, target, _ = batch

        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward pass
        output = self(images)

        # individual losses per partitioning
        losses = [
            self.losses[i](output[i], target[i])
            for i in range(len(output))
        ]

        loss = sum(losses)

        # stats
        losses_stats = {
            f"loss_train/{p}": l
            for (p, l) in zip([p.shortname for p in self.partitionings], losses)
        }
        for metric_name, metric_value in losses_stats.items():
            self.log(metric_name, metric_value, prog_bar=True, logger=True)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, **losses_stats}

    def validation_step(self, batch, batch_idx):
        images, target, true_lats, true_lngs, _ = batch

        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward
        output = self(images)

        # loss calculation
        losses = [
            self.losses[i](output[i], target[i])
            for i in range(len(output))
        ]

        loss = sum(losses)

        # log top-k accuracy for each partitioning
        individual_accuracy_dict = utils_global.accuracy(
            output, target, [p.shortname for p in self.partitionings]
        )
        # log loss for each partitioning
        individual_loss_dict = {
            f"loss_val/{p}": l
            for (p, l) in zip([p.shortname for p in self.partitionings], losses)
        }

        # log GCD error@km threshold
        distances_dict = utils_global.get_distances(output, true_lats, true_lngs, self.partitionings, self.hierarchy)

        output = {
            "loss_val/total": loss,
            **individual_accuracy_dict,
            **individual_loss_dict,
            **distances_dict,
        }
        return output

    def validation_epoch_end(self, outputs):
        pnames = [p.shortname for p in self.partitionings]

        # top-k accuracy and loss per partitioning
        loss_acc_dict = utils_global.summarize_loss_acc_stats(pnames, outputs)

        # GCD stats per partitioning
        gcd_dict = utils_global.summarize_gcd_stats(pnames, outputs, self.hierarchy)

        metrics = {
            "val_loss": loss_acc_dict["loss_val/total"],
            **loss_acc_dict,
            **gcd_dict,
        }
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, logger=True)

    def _multi_crop_inference(self, batch):
        images, meta_batch = batch
        cur_batch_size = images.shape[0]
        ncrops = images.shape[1]

        # reshape crop dimension to batch
        images = torch.reshape(images, (cur_batch_size * ncrops, *images.shape[2:]))

        # forward pass
        yhats = self(images)
        yhats = [torch.nn.functional.softmax(yhat, dim=1) for yhat in yhats]

        # respape back to access individual crops
        yhats = [
            torch.reshape(yhat, (cur_batch_size, ncrops, *list(yhat.shape[1:])))
            for yhat in yhats
        ]

        # calculate max over crops
        yhats = [torch.max(yhat, dim=1)[0] for yhat in yhats]

        hierarchy_preds = None
        if self.hierarchy is not None:
            hierarchy_logits = torch.stack(
                [yhat[:, self.hierarchy.M[:, i]] for i, yhat in enumerate(yhats)],
                dim=-1,
            )
            hierarchy_preds = torch.prod(hierarchy_logits, dim=-1)

        return yhats, meta_batch, hierarchy_preds

    def inference(self, batch):

        yhats, meta_batch, hierarchy_preds = self._multi_crop_inference(batch)

        if self.hierarchy is not None:
            nparts = len(self.partitionings) + 1
        else:
            nparts = len(self.partitionings)

        pred_class_dict = {}
        pred_lat_dict = {}
        pred_lng_dict = {}
        for i in range(nparts):
            # get pred class indices
            if self.hierarchy is not None and i == len(self.partitionings):
                pname = "hierarchy"
                pred_classes = torch.argmax(hierarchy_preds, dim=1)
                i = i - 1
            else:
                pname = self.partitionings[i].shortname
                pred_classes = torch.argmax(yhats[i], dim=1)

            # calculate GCD
            pred_lats, pred_lngs = map(
                list,
                zip(
                    *[
                        self.partitionings[i].get_lat_lng(c)
                        for c in pred_classes.tolist()
                    ]
                ),
            )
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)
            pred_lat_dict[pname] = pred_lats
            pred_lng_dict[pname] = pred_lngs
            pred_class_dict[pname] = pred_classes

        return meta_batch["img_path"], pred_class_dict, pred_lat_dict, pred_lng_dict

    def test_step(self, batch, batch_idx, dataloader_idx=None):

        yhats, meta_batch, hierarchy_preds = self._multi_crop_inference(batch)

        distances_dict = {}
        if self.hierarchy is not None:
            nparts = len(self.partitionings) + 1
        else:
            nparts = len(self.partitionings)

        for i in range(nparts):
            # get pred class indices
            if self.hierarchy is not None and i == len(self.partitionings):
                pname = "hierarchy"
                pred_classes = torch.argmax(hierarchy_preds, dim=1)
                i = i - 1
            else:
                pname = self.partitionings[i].shortname
                pred_classes = torch.argmax(yhats[i], dim=1)

            # calculate GCD
            pred_lats, pred_lngs = map(
                list,
                zip(
                    *[
                        self.partitionings[i].get_lat_lng(c)
                        for c in pred_classes.tolist()
                    ]
                ),
            )
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)

            distances = utils_global.vectorized_gc_distance(
                pred_lats,
                pred_lngs,
                meta_batch["latitude"].type_as(pred_lats),
                meta_batch["longitude"].type_as(pred_lngs),
            )
            distances_dict[pname] = distances

        return distances_dict

    def test_epoch_end(self, outputs):
        result = utils_global.summarize_test_gcd(
            [p.shortname for p in self.partitionings], outputs, self.hierarchy
        )
        return {**result}

    def configure_optimizers(self):
        optim_args = get_optimizer_parameters(self.hparams.optim, self)
        optim_feature_extractor = torch.optim.SGD(optim_args)

        return {
            "optimizer": optim_feature_extractor,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optim_feature_extractor, **self.hparams.scheduler["params"]
                ),
                "interval": "epoch",
                "name": "lr",
            },
        }

    def train_dataloader(self):

        with open(self.hparams.train_label_mapping, "r") as f:
            target_mapping = json.load(f)

        tfm = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomResizedCrop(self.image_dim, scale=(0.66, 1.0)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        cache_size = self.hparams.get("cache_size", 350000)
        logging.info(f"Using cache_size {cache_size}")
        dataset = MsgPackIterableDatasetMultiTargetWithDynLabels(
            path=self.hparams.msgpack_train_dir,
            target_mapping=target_mapping,
            key_img_id=self.hparams.key_img_id,
            key_img_encoded=self.hparams.key_img_encoded,
            shuffle=True,
            cache_size=cache_size,
            transformation=tfm,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers_per_loader,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):

        with open(self.hparams.val_label_mapping, "r") as f:
            target_mapping = json.load(f)

        tfm = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(self.image_dim),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        dataset = MsgPackIterableDatasetMultiTargetWithDynLabels(
            path=self.hparams.msgpack_val_dir,
            target_mapping=target_mapping,
            key_img_id=self.hparams.key_img_id,
            key_img_encoded=self.hparams.key_img_encoded,
            shuffle=False,
            transformation=tfm,
            meta_path=self.hparams.val_meta_path,
            cache_size=1024,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers_per_loader,
            pin_memory=True,
        )

        return dataloader


def parse_args():
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path)
    args.add_argument("--progbar", action="store_true")
    return args.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Use OmegaConf for yaml interpolation and inheritance
    config = OmegaConf.load(args.config)
    model_params = OmegaConf.to_object(config["model_params"])
    trainer_params = OmegaConf.to_object(config["trainer_params"])
    
    name = Path(model_params["arch"]) / model_params.get("name", "")
    out_dir = Path(config["out_dir"]) / name

    # Init classifier
    model = MultiPartitioningClassifier(hparams=model_params)
    logger = pl.loggers.wandb.WandbLogger(name=str(name).replace("/", "_"), project="geoestimation") 
    checkpointer = pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=out_dir / "ckpts", filename="{epoch:03d}-{val_loss:.4f}")

    trainer = pl.Trainer(
        **trainer_params,
        logger=logger,
        val_check_interval=model_params["val_check_interval"],
        callbacks=[checkpointer],
    )

    trainer.fit(model)

if __name__ == "__main__":
    main()
