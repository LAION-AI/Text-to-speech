from typing import Any, Union, Optional
import torch
import argparse
from os import path as osp
from functools import partial
from denoiser.audio import Audioset

from .common import Base
from . import audio as audio_ops
from .denoiser_utils import get_model
from utils.helpers import exists
from config import settings

cache_dir = osp.join(settings.CACHE_DIR, "weights", "enhancement")


class DenoiseAudio(Base):
    MODEL_CHOICES = {
        "meta_denoiser_master64": {
            "model": partial(
                get_model,
                args=argparse.Namespace(
                    **{
                        "model_path": None,
                        "hub_dir": osp.join(cache_dir, "fair-denoiser-master64"),
                        "master64": True,
                        "dns64": False,
                        "valentini": False,
                        "dns48": False,
                    }
                ),
            ),
            "target": "enhance_with_denoiser",
        },
        "meta_denoiser_dns64": {
            "model": partial(
                get_model,
                args=argparse.Namespace(
                    **{
                        "model_path": None,
                        "hub_dir": osp.join(cache_dir, "meta-denoiser-dns64"),
                        "master64": False,
                        "dns64": True,
                        "valentini": False,
                        "dns48": False,
                    }
                ),
            ),
            "target": "enhance_with_denoiser",
        },
        "meta_denoiser_valentini": {
            "model": partial(
                get_model,
                args=argparse.Namespace(
                    **{
                        "model_path": None,
                        "hub_dir": osp.join(cache_dir, "meta-denoiser-valentini"),
                        "master64": False,
                        "dns64": False,
                        "valentini": True,
                    }
                ),
            ),
            "target": "enhance_with_denoiser",
        },
        "meta_denoiser_dns48": {
            "model": partial(
                get_model,
                args=argparse.Namespace(
                    **{
                        "model_path": None,
                        "hub_dir": osp.join(cache_dir, "meta-denoiser-dns48"),
                        "master64": False,
                        "dns64": False,
                        "valentini": False,
                        "dns48": True,
                    }
                ),
            ),
            "target": "enhance_with_denoiser",
        },
    }

    def _init_(
        self,
        model_choice: str,
        sampling_rate: int = 16000,
        padding: Union[bool,str] = True,
        max_length: Union[int,None] = None,
        pad_to_multiple_of: Union[int,None] = None,
        max_audio_len: int = 5,
        dry=0,
        **kwargs,
    ) -> None:
        super()._init_(
            model_choice,
            sampling_rate,
            padding,
            max_length,
            pad_to_multiple_of,
            max_audio_len,
            **kwargs,
        )
        self.dry = dry

    def enhance_with_denoiser(self, audio_path, save_to_file=False, **kwargs):
        metadata = [(audio_path, audio_ops.get_audio_info(audio_path))]
        dataset = Audioset(
            metadata,
            with_path=False,
            sample_rate=self.model["model"].sample_rate,
            channels=self.model["model"].chin,
            convert=True,
        )
        signal = dataset[0]
        with torch.no_grad():
            estimate = self.model["model"](signal.cuda())
            # estimate = (1 - self.dry) * estimate + self.dry * signal
        if save_to_file:
            save_dir = kwargs.get("save_dir")
            enhanced_audio = estimate.detach().cpu().squeeze(0)
            denoised_path = self.save_to_file(
                enhanced_audio, sr=16000, save_dir=save_dir
            )
        return denoised_path

    def predict(self, audio_path, **kwargs) -> torch.Tensor:
        if hasattr(self.model, "enhance_file"):
            enhanced_audio = self.model.enhance_file(audio_path)
        else:
            raise NotImplementedError(
                f"{self.model_choice} doesn't have any supported methods"
            )
        if enhanced_audio.ndim == 3:
            enhanced_audio = enhanced_audio.squeeze(-1)
        return enhanced_audio

    def _call_(
        self,
        audio_path: str = None,
        audio: torch.Tensor = None,
        save_to_file=False,
        offload=False,
        **kwargs,
    ) -> Any:
        assert exists(audio_path) or exists(
            audio
        ), "Either audio_path or audio tensor is required"



        if isinstance(self.model, dict):
            enhanced_audio = self.MODEL_CHOICES[self.model_choice]["target"](
                audio_path=audio_path, audio=audio, **kwargs
            )
        else:
            enhanced_audio = self.predict(audio_path=audio_path, audio=audio, **kwargs)

        if save_to_file:
            enhanced_audio = enhanced_audio.detach().cpu()
            enhanced_audio = self.save_to_file(
                enhanced_audio, sr=audio_info.sample_rate, save_dir=save_dir
            )
        elif offload:
            enhanced_audio = enhanced_audio.detach().cpu()
        return enhanced_audio