#!/bin/bash
set -e

#
# This script downloads the 'dataset' and 'weights' folders from
# geolocation_via_guidebook_grounding.berkeleyvision.org
#

URL_BASE="http://geolocation_via_guidebook_grounding.berkeleyvision.org/"

# Get the directory of this script so that we can reference paths correctly no matter which folder
# the script was launched from:
SCRIPTS_DIR="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname ${SCRIPTS_DIR})"
PROJ_DIR="$(dirname ${PROJ_DIR})/g3"
echo "SCRIPTS_DIR ${SCRIPTS_DIR}"
echo "PROJ_DIR ${PROJ_DIR}"

# Create data_dir
DATA_DIR="${PROJ_DIR}/dataset"
mkdir -p "${DATA_DIR}/features"
mkdir -p "${DATA_DIR}/loss_weight"
mkdir -p "${DATA_DIR}/pseudo_labels"
mkdir -p "${DATA_DIR}/s2_cells"
mkdir -p "${DATA_DIR}/test"
mkdir -p "${DATA_DIR}/train"
mkdir -p "${DATA_DIR}/val"

# Download files necessary to get panoramas and panocutting:
pushd "${DATA_DIR}"
echo "Downloading project data files to ${DATA_DIR}..."
curl -o ./features/guidebook_roberta_base.pkl "${URL_BASE}/dataset/features/guidebook_roberta_base.pkl"
curl -o ./features/news_roberta_base.pkl "${URL_BASE}/dataset/features/news_roberta_base.pkl"
curl -o ./features/streetview_clip_rn50x16.pkl "${URL_BASE}/dataset/features/streetview_clip_rn50x16.pkl"
curl -o ./loss_weight/countries.json "${URL_BASE}/dataset/loss_weight/countries.json"
curl -o ./pseudo_labels/countries.json "${URL_BASE}/dataset/pseudo_labels/countries.json"
curl -o ./s2_cells/countries.json "${URL_BASE}/dataset/s2_cells/countries.json"
curl -o ./test/test.csv "${URL_BASE}/dataset/test/test.csv"
curl -o ./train/train.csv "${URL_BASE}/dataset/train/train.csv"
curl -o ./val/val.csv "${URL_BASE}/dataset/val/val.csv"
curl -o ./guidebook.json "${URL_BASE}/dataset/guidebook.json"
popd


## Download weights
WEIGHTS_DIR="${PROJ_DIR}/weights"
mkdir -p "${WEIGHTS_DIR}/g3/132/ckpts"
pushd "${WEIGHTS_DIR}"
echo "Downloading model weights to ${WEIGHTS_DIR}..."
curl -o ./g3/132/ckpts/last.ckpt "${URL_BASE}/weights/g3/132/ckpts/last.ckpt"
curl -o ./g3/132/config.json "${URL_BASE}/weights/g3/132/config.json"
curl -o ./isn_epoch=014-val_loss=18.4833.ckpt   "${URL_BASE}/weights/isn_epoch=014-val_loss=18.4833.ckpt"
popd
