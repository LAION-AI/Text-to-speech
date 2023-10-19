from os import path as osp
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig, ListConfig
import json
import requests
from .validators import uri_validator


def load_configs(configs):
    if isinstance(configs, (DictConfig, ListConfig)):
        return configs

    if isinstance(configs, list):
        config = OmegaConf.merge(*[OmegaConf.load(conf) for conf in configs]) if osp.exists(configs[0]) else OmegaConf.create(configs)
    else:
        config = OmegaConf.load(configs) if osp.exists(configs) else OmegaConf.create(configs)
        
    return config

def merge_configs(*confs):
    return OmegaConf.merge(*[load_configs(conf) for conf in confs])

def load_metadata(metadata_path):
    if not osp.isfile(metadata_path):
        metadata_path = osp.join(metadata_path, "metadata.jsonl")
        
    if not osp.isfile(metadata_path):
        raise FileNotFoundError(f"'metadata.jsonl' file missing in '{metadata_path}'")
        
    with open(metadata_path, "r") as file:
        ext = osp.splitext(metadata_path)[-1]
        return json.load(file) if ext == ".json" else [json.loads(line) for line in file.read().splitlines()]

def save_metadata(metadata, save_path):
    parent_dir = Path(osp.dirname(save_path))
    parent_dir.mkdir(exist_ok=True, parents=True)
    
    with open(save_path, "w") as f:
        if save_path.endswith(".jsonl"):
            f.writelines(f"{json.dumps(entry)}\n" for entry in metadata)
        elif save_path.endswith(".json"):
            json.dump(metadata, f)
        elif save_path.endswith(".txt"):
            f.write("\n".join(metadata) if isinstance(metadata, list) else metadata)
        else:
            raise NotImplementedError(f"{osp.splitext(save_path)[-1]} is not supported...")

def download_file_from_url(url, save_path, show_progress=True):
    if not uri_validator(url):
        raise ValueError(f"'{url}' doesn't seem to be a valid url")

    parent_dir = Path(osp.dirname(save_path))
    parent_dir.mkdir(exist_ok=True, parents=True)

    with open(save_path, "wb") as handle:
        response = requests.get(url, stream=True, timeout=20)
        if response.status_code != 200:
            raise AssertionError(f"Couldn't download the file, the url returned status code: {response.status_code}")

        for data in tqdm(response.iter_content(), disable=not show_progress):
            handle.write(data)