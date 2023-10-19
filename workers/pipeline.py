import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import argparse
from omegaconf import OmegaConf

from utils.io import load_configs
from utils.helpers import get_obj_from_str


def run(configs):
    config = load_configs(configs)["pipeline"]
    downloader = get_obj_from_str(config["loader"]["target"])(
        **config["loader"]["args"]
    )
    manager = get_obj_from_str(config["manager"]["target"])(
        configs=OmegaConf.to_yaml(config)
    )
    save_dir = None
    for processor in config["processors"]:
        if processor["name"] == "downloader":
            save_dir = processor["args"]["save_dir"]
    for file_metadata in downloader.walk_files(save_dir=save_dir):
        manager(file_metadata=file_metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "TTS audio pipeline",
        description="Run data workers for preparing tts audio datasets",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Config file path(s) for pipeline orchestration"
        "Configs will be merged from left to right",
    )
    args = parser.parse_args()

    run(args.configs)
