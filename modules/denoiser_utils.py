import logging

import torch.hub
from denoiser.demucs import Demucs
from denoiser.utils import deserialize_model

from .common import logger


ROOT = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/"
DNS_48_URL = ROOT + "dns48-11decc9d8e3f0998.th"
DNS_64_URL = ROOT + "dns64-a7761ff99a7d5bb6.th"
MASTER_64_URL = ROOT + "master64-8a5dfb4bb92753dd.th"
VALENTINI_NC = ROOT + "valentini_nc-93fc4337.th"  # Non causal Demucs on Valentini


def _demucs(pretrained, url, model_dir=None, **kwargs):
    model = Demucs(**kwargs, sample_rate=16_000)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url, model_dir=model_dir, map_location="cpu"
        )
        model.load_state_dict(state_dict)
    return model


def dns48(pretrained=True, model_dir=None):
    return _demucs(pretrained, DNS_48_URL, hidden=48, model_dir=model_dir)


def dns64(pretrained=True, model_dir=None):
    return _demucs(pretrained, DNS_64_URL, hidden=64, model_dir=model_dir)


def master64(pretrained=True, model_dir=None):
    return _demucs(pretrained, MASTER_64_URL, hidden=64, model_dir=model_dir)


def valentini_nc(pretrained=True, model_dir=None):
    return _demucs(
        pretrained,
        VALENTINI_NC,
        hidden=64,
        causal=False,
        stride=2,
        resample=2,
        model_dir=model_dir,
    )


def add_model_flags(parser):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-m", "--model_path", help="Path to local trained model.")
    group.add_argument(
        "--hub_dir", default=None, help="Path to save torch/hf hub models."
    )
    group.add_argument(
        "--dns48",
        action="store_true",
        help="Use pre-trained real time H=48 model trained on DNS.",
    )
    group.add_argument(
        "--dns64",
        action="store_true",
        help="Use pre-trained real time H=64 model trained on DNS.",
    )
    group.add_argument(
        "--master64",
        action="store_true",
        help="Use pre-trained real time H=64 model trained on DNS and Valentini.",
    )
    group.add_argument(
        "--valentini_nc",
        action="store_true",
        help="Use pre-trained H=64 model trained on Valentini, non causal.",
    )


def get_model(args):
    """
    Load local model package or torchhub pre-trained model.
    """
    if args.model_path:
        logger.info("Loading model from %s", args.model_path)
        pkg = torch.load(args.model_path, "cpu")
        if "model" in pkg:
            if "best_state" in pkg:
                pkg["model"]["state"] = pkg["best_state"]
            model = deserialize_model(pkg["model"])
        else:
            model = deserialize_model(pkg)
    elif args.dns64:
        logger.info("Loading pre-trained real time H=64 model trained on DNS.")
        model = dns64(args.hub_dir)
    elif args.master64:
        logger.info(
            "Loading pre-trained real time H=64 model trained on DNS and Valentini."
        )
        model = master64(args.hub_dir)
    elif args.valentini_nc:
        logger.info("Loading pre-trained H=64 model trained on Valentini.")
        model = valentini_nc(args.hub_dir)
    else:
        logger.info("Loading pre-trained real time H=48 model trained on DNS.")
        model = dns48(args.hub_dir)
    logger.debug(model)
    return model
