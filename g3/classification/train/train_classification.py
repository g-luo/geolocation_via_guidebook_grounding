import io
import json
import logging
import os
import pickle
import pytorch_lightning as pl
import sys
import torch
import torch.nn as nn
import wandb

from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from datetime import datetime
from omegaconf import OmegaConf
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from setproctitle import setproctitle
from typing import Any, Dict, cast

sys.path.append("../")
from classification import utils_global
from classification.layers import SimpleLinearProjectionAttention
from classification.train.train_base import MultiPartitioningBase

class MultiPartitioningClassifier(MultiPartitioningBase):
    def __init__(self, hparams: Namespace, wandb_logger: WandbLogger):
        self.wandb_logger: WandbLogger = wandb_logger
        self.setting = hparams.get("setting", "image")
        logging.info(f"Using setting {self.setting}")

        if hparams.get("text_features_file", ""):
            self.text_features = pickle.load(open(hparams["text_features_file"], "rb"))
            self.text_features = [torch.from_numpy(t) for t in self.text_features.values()]
            self.text_features = utils_global.vstack(self.text_features)
            # self.text_features = self.text_features.float()
            logging.info(f"Loading text features from {hparams['text_features_file']}")
            logging.info(f"len(text_features): {len(self.text_features):,}")
        else:
            self.text_features = None

        if hparams.get("image_features_file", ""):
            self.image_features = pickle.load(open(hparams["image_features_file"], "rb"))
            self.image_features = {
                _id: torch.from_numpy(img_feat) for _id, img_feat in self.image_features.items()
            }
            logging.info(f"len(image_features): {len(self.image_features):,}")
        else:
            self.image_features = None
            logging.info(f"Using same image embedding for classification and attention")

        super().__init__(hparams)

        if hparams.get("text_labels_file", ""):
            self.alpha = hparams.get("attn_loss_alpha", 0.0)
            self.text_labels = json.load(open(hparams["text_labels_file"]))
            self.attn_loss_type = self.hparams.get("attn_loss_type", None)
            # Reweight BCE loss due to imbalance of text importance
            # Some pieces of text may never correspond to a GT image sample
            self.attn_loss_weight = hparams.get("attn_loss_weight", 1)
            self.bce_losses = [torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.attn_loss_weight)) for _ in range(len(self.text_features))]
            logging.info(f"Adding text labels from {hparams['text_labels_file']}")
            logging.info(
                f"Adding attention loss with alpha = {self.alpha}, loss type = {self.attn_loss_type}, attn_loss_weight = {self.attn_loss_weight}"
            )
        else:
            self.text_labels = None
            self.alpha = 0
            self.bce_losses = []
            logging.info("Not adding attention loss")

        # Reweight CE loss due to country class imbalance
        if hparams.get("loss_weight", ""):
            loss_weights = json.load(open(hparams["loss_weight"]))
            self.losses = [torch.nn.CrossEntropyLoss(weight=torch.tensor(loss_weight)) for loss_weight in loss_weights]
        else:
            self.losses = [torch.nn.CrossEntropyLoss() for _ in range(len(self.partitionings))]

    def build_model(self):
        model, nfeatures_img, image_dim = utils_global.build_base_model(self.hparams.arch)    
        nfeatures_attn_text = self.hparams.get("attn_text_size", 0)
        nfeatures_attn_image = self.hparams.get("attn_image_size", 0)

        nfeatures = 0
        if self.setting == "image":
            nfeatures = nfeatures_img
        elif self.setting == "clip":
            nfeatures =  nfeatures_attn_image
        elif self.setting == "image_and_clip":
            nfeatures = nfeatures_img + nfeatures_attn_image
        elif self.setting == "clip_and_clues":
            nfeatures =  nfeatures_attn_image + nfeatures_attn_text
        elif self.setting == "image_and_clues":
            nfeatures = nfeatures_img + nfeatures_attn_text
        elif self.setting == "image_clip_clues":
            nfeatures = nfeatures_img + nfeatures_attn_image + nfeatures_attn_text
        else:
            raise NotImplementedError(f"Unsupported value for 'setting': {self.setting}")

        if "clues" in self.setting:
            attn_layer = SimpleLinearProjectionAttention(
                nfeatures_attn_image,
                len(self.text_features),
                norm_type=self.hparams.get("attn_input_norm_type", "batch_norm"),
                beta=self.hparams.get("attn_beta", 1.0),
            )
        else:
            attn_layer = None

        # Build the classifier
        classifier_use_bn = self.hparams.get("classifier_use_bn", True)
        classifier = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.BatchNorm1d(nfeatures) if classifier_use_bn else nn.Identity(),
                    torch.nn.Linear(nfeatures, len(self.partitionings[i])),
                )
                for i in range(len(self.partitionings))
            ]
        )

        if self.hparams.weights:
            logging.info("Load weights from pre-trained model")
            model, classifier, attn_layer = utils_global.load_weights_if_available(
                model, classifier, attn_layer, self.hparams.weights
            )

        self.model = model
        self.classifier = classifier
        self.attn_layer = attn_layer
        return image_dim

    def get_attn_labels(self, ids, i):
        attn_labels = torch.zeros((len(ids), len(self.text_features)))
        for _id_i, _id in enumerate(ids):
            attn_labels[_id_i, self.text_labels[_id][i]] = 1
        return attn_labels.to(self.device)

    def get_attention(self, batch, text_embedding, image_embedding, ids):
        assert self.attn_layer is not None, f"self.attn_layer should not be None!"
        attn_logits = self.attn_layer(image_embedding)
        extra = {}
        extra["attn_logits"] = attn_logits

        if self.text_labels:
            attn_labels = self.get_attn_labels(ids, 0)
            extra["attn_labels"] = attn_labels

        if self.hparams.get("attn_relu", True):
            attn_logits = torch.nn.functional.relu(attn_logits)

        normalization_type = self.hparams.get("attn_layer_norm_type", "sigmoid")
        if normalization_type == "sigmoid":
            attn_scores = torch.nn.functional.sigmoid(attn_logits)
        elif normalization_type == "softmax":
            attn_scores = torch.nn.functional.softmax(attn_logits, dim=1)
        elif normalization_type == "relu":
            attn_scores = torch.nn.functional.relu(attn_logits)
        else:
            raise NotImplementedError

        extra["attn_scores"] = attn_scores
        self.log("train/attn_score_min", attn_scores.min().detach().cpu())
        self.log("train/attn_score_max", attn_scores.max().detach().cpu())
        weighted_text_embedding = torch.matmul(attn_scores, text_embedding)
        return weighted_text_embedding, extra

    def get_image_embedding(self, batch):
        image_embedding = self.model(batch)
        return image_embedding

    def get_clip_embedding(self, batch, ids):
        clip_embedding = torch.stack([self.image_features[_id] for _id in ids])
        clip_embedding = clip_embedding.to(self.device)
        return clip_embedding

    def get_text_embedding(self, batch, ids, image_embedding):
        if self.image_features:
            image_embedding = self.get_clip_embedding(batch, ids)
        self.text_features = self.text_features.to(self.device)
        image_embedding = image_embedding.to(self.dtype)
        self.text_features = self.text_features.to(self.dtype)
        return self.get_attention(batch, self.text_features, image_embedding, ids)

    def forward(self, x):
        output = {}
        batch, ids = x

        if self.setting == "image":
            fv = self.get_image_embedding(batch)
        elif self.setting == "clip":
            clip_embedding = self.get_clip_embedding(batch, ids)
            fv = clip_embedding
        elif self.setting == "image_and_clip":
            image_embedding = self.get_image_embedding(batch)
            clip_embedding = self.get_clip_embedding(batch, ids)
            fv = torch.cat([image_embedding, clip_embedding], dim=-1)
        elif self.setting == "image_and_clues":
            image_embedding = self.get_image_embedding(batch)
            text_embedding, extra = self.get_text_embedding(batch, ids, image_embedding)
            output["attn"] = extra
            fv = torch.cat([image_embedding, text_embedding], dim=-1)
        elif self.setting == "clip_and_clues":
            clip_embedding = self.get_clip_embedding(batch, ids)
            text_embedding, extra = self.get_text_embedding(batch, ids, image_embedding)
            output["attn"] = extra
            fv = torch.cat([text_embedding, clip_embedding], dim=-1)
        elif self.setting == "image_clip_clues":
            image_embedding = self.get_image_embedding(batch)
            clip_embedding = self.get_clip_embedding(batch, ids)
            text_embedding, extra = self.get_text_embedding(batch, ids, image_embedding)
            output["attn"] = extra
            fv = torch.cat([image_embedding, text_embedding, clip_embedding], dim=-1)
        else:
            raise NotImplementedError

        yhats = [self.classifier[i](fv) for i in range(len(self.partitionings))]
        output["output"] = yhats

        # If doing feature extraction, clear the features after each fwd pass:
        # if hasattr(self, "features"):
        #     for k in self.features:
        #         del self.features[k]
        return output

    def get_losses(self, output, target, ids, dtype):
        losses_stats = {}
        losses = []
        for i in range(len(output["output"])):
            p = self.partitionings[i].shortname
            if self.losses[i].weight is not None:
                self.losses[i].weight = self.losses[i].weight.to(self.device)
            ce_loss = self.losses[i](output["output"][i], target[i])
            losses_stats[f"loss_{dtype}/{p}/cls_loss"] = ce_loss

            if self.text_labels and self.alpha > 0:
                # Add BCEWithLogitsLoss for the one hot vector of
                # if a clue should be attended to or not
                self.bce_losses[i].pos_weight = self.bce_losses[i].pos_weight.to(self.device)
                attn_labels = output["attn"]["attn_labels"]
                if self.attn_loss_type == "bce":
                    attn_loss = self.bce_losses[i](
                        output["attn"]["attn_logits"], attn_labels.float()
                    )
                else:
                    raise NotImplementedError
                losses_stats[f"loss_{dtype}/{p}/attn_loss"] = attn_loss
                total_loss = ce_loss * (1 - self.alpha) + attn_loss * self.alpha
            else:
                total_loss = ce_loss
            losses_stats[f"loss_{dtype}/{p}"] = total_loss
            losses.append(total_loss)
        return sum(losses), losses_stats

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        images, target, _id = batch

        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        output = self((images, _id))
        loss, losses_stats = self.get_losses(output, target, _id, "train")
        for metric_name, metric_value in losses_stats.items():
            self.log(metric_name, metric_value, prog_bar=True, logger=True)
        return {"loss": loss, **losses_stats}

    def validation_step(self, batch, batch_idx):
        images, target, true_lats, true_lngs, _id = batch

        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward
        output = self((images, _id))
        loss, individual_loss_dict = self.get_losses(output, target, _id, "val")

        # log top-k accuracy for each partitioning
        individual_accuracy_dict = utils_global.accuracy(
            output["output"], target, [p.shortname for p in self.partitionings]
        )

        labels_dict = {"labels": target, "ids": _id}
        predictions_dict = {"predictions": [o.argmax(dim=-1) for o in output["output"]]}

        if "attn" in output:
            attn_dict = {"attn": output["attn"]}
        else:
            attn_dict = {}

        # log GCD error@km threshold
        distances_dict = utils_global.get_distances(
            output["output"], true_lats, true_lngs, self.partitionings, self.hierarchy
        )
        output = {
            "loss_val/total": loss,
            **individual_accuracy_dict,
            **individual_loss_dict,
            **distances_dict,
            **labels_dict,
            **predictions_dict,
            **attn_dict,
        }
        return output

    def validation_epoch_end(self, outputs):
        pnames = [p.shortname for p in self.partitionings]
        loss_acc_dict = utils_global.summarize_loss_acc_stats(pnames, outputs)
        gcd_dict = utils_global.summarize_gcd_stats(pnames, outputs, self.hierarchy)

        for i in range(len(self.partitionings)):
            attn_acc_dict = {}
            if self.text_labels:
                # Get binary classification acc for if clue should be included or not
                attn_labels = torch.vstack([o["attn"]["attn_labels"] for o in outputs])
                attn_predictions = torch.vstack([o["attn"]["attn_scores"] > 0.5 for o in outputs])
                attn_acc = (attn_labels == attn_predictions).float()
                attn_pos_acc = attn_acc[attn_labels == 1].mean()
                attn_neg_acc = attn_acc[attn_labels == 0].mean()
                attn_acc_dict["attn_acc/pos_class"] = attn_pos_acc
                attn_acc_dict["attn_acc/neg_class"] = attn_neg_acc
                attn_acc_dict["attn_num/pos_class"] = (attn_labels == 1).sum()
                attn_acc_dict["attn_num/neg_class"] = (attn_labels == 0).sum()
                attn_acc_dict["attn_acc/avg_class"] = (attn_pos_acc + attn_neg_acc) / 2

        metrics = {
            "val_loss": loss_acc_dict["loss_val/total"],
            **loss_acc_dict,
            **gcd_dict,
            **attn_acc_dict,
        }
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, logger=True)


def parse_args():
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path)
    args.add_argument("--progbar", action="store_true")
    args.add_argument(
        "overrides",
        nargs="*",
        help="Any key=svalue arguments to override config values "
        "(use dots for.nested=overrides)",
    )
    return args.parse_args()

def load_yaml(f):
    abs_f = os.path.abspath(f)
    mapping = OmegaConf.load(f)
    includes = mapping.get("includes", [])
    include_mapping = OmegaConf.create()

    for include in includes:
        # Load includes relative to f
        include = os.path.join(os.path.dirname(f), include)
        current_include_mapping = load_yaml(include)
        include_mapping = OmegaConf.merge(include_mapping, current_include_mapping)

    mapping.pop("includes", None)
    mapping = OmegaConf.merge(include_mapping, mapping)
    return mapping

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Use OmegaConf for yaml interpolation and inheritance
    config = load_yaml(args.config)
    cli_overrides = OmegaConf.from_cli(args.overrides)
    config = OmegaConf.merge(config, cli_overrides)

    model_params = OmegaConf.to_object(config["model_params"])
    trainer_params = OmegaConf.to_object(config["trainer_params"])
    name = model_params.get("name", "")
    out_dir = Path(config["out_dir"]) / name / f"{config['seed']}"
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(Path(out_dir) / "config.yaml", "w") as fp:
        OmegaConf.save(config=config, f=fp)
    config_artifact = wandb.Artifact("config_file", "file")
    config_artifact.add_file(Path(out_dir) / "config.yaml")
    seed_everything(config["seed"])

    # Init classifier
    logger = WandbLogger(
        name=f"{name}-seed_{config['seed']}".replace("/", "_"),
        project=config["wandb_dir"],
        config=OmegaConf.to_object(config),
    )
    logger.experiment.log_artifact(config_artifact)
    model = MultiPartitioningClassifier(hparams=model_params, wandb_logger=logger)
    checkpointer = ModelCheckpoint(
        dirpath=out_dir / "ckpts",
        filename="epoch{epoch:03d}-acc1_val_countries{acc1_val/countries:.4f}",
        save_top_k=1,
        monitor="acc1_val/countries",
        mode="max",
        verbose=True,
        auto_insert_metric_name=False,
        save_last=True
    )
    trainer = pl.Trainer(
        **trainer_params,
        logger=logger,
        val_check_interval=model_params["val_check_interval"],
        callbacks=[checkpointer],
        accelerator="gpu",
    )

    print(model)
    seed_everything(config["seed"])
    trainer.fit(model)


if __name__ == "__main__":
    setproctitle(Path(__file__).name)
    main()
