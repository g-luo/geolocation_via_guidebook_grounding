from argparse import ArgumentParser
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def infer_country(dataset, pseudo_labels, guidebook, country_to_id):
  label_mapping, name_mapping = {}, {}
  for img_id in tqdm(sorted(dataset["IMG_ID"].to_list())):
    clue_idxs = pseudo_labels[img_id][0]
    clue_countries = [guidebook[idx]["geoparsed"] for idx in clue_idxs]
    for countries in clue_countries:
      unique_countries = set([c["Country"] for c in countries])
      if len(unique_countries) == 1:
        # Continue to next img_id after we find a match
        country = unique_countries.pop()
        label_mapping[img_id] = [country_to_id[country]]
        name_mapping[img_id] = [country]
        break
    if img_id not in label_mapping:
      # Handle the special case where all clues that mention Reunion co-occur with another country
      label_mapping[img_id] = [187]
      name_mapping[img_id] = ["Reunion"]
  return label_mapping, name_mapping

def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--output",
        type=Path,
    )
    args.add_argument(
        "--url_csv",
        type=Path,
    )
    args.add_argument(
        "--s2_cells_csv",
        type=Path,
    )
    args.add_argument(
        "--pseudo_labels",
        type=Path,
    )
    args.add_argument(
        "--guidebook",
        type=Path,
    )
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(args.url_csv)
    country_to_id = pd.read_csv(args.s2_cells_csv, skiprows=0)
    country_to_id = {c["country"]: c["class_label"] for c in country_to_id.to_dict(orient="records")}
    
    pseudo_labels = json.load(open(args.pseudo_labels))
    guidebook = json.load(open(args.guidebook))

    label_mapping, name_mapping = infer_country(dataset, pseudo_labels, guidebook, country_to_id)
    json.dump(label_mapping, open(args.output / "countries.json", "w"))
    json.dump(name_mapping, open(args.output / "countries_names.json", "w"))