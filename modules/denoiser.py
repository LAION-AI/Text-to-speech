from typing import Any
import torch
import torchaudio
import argparse
import os
import time
from functools import partial
from os import path as osp
from denoiser.audio import Audioset
from .common import Base
from . import audio_ops
from .denoiser_utils import get_model
from utils.helpers import exists
from config import settings


class DenoiseAudio(Base):

    def __init__(self, model_choice: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = self.load_model(model_choice, **kwargs)

    @staticmethod
    def load_model(model_choice: str, cache_dir=None, **kwargs) -> dict:
        hub_dir = osp.join(cache_dir or settings.CACHE_DIR, "weights", "denoiser", model_choice)
        args = {**kwargs, "model_path": None, "hub_dir": hub_dir}

        for key in ["master64", "dns64", "valentini", "dns48"]:
            args[key] = model_choice.endswith(key)

        return {
            "model": partial(get_model, args=argparse.Namespace(**args)),
            "target": "apply_denoiser",
        }

    def apply_denoiser(self, audio_path, **kwargs):
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
            estimate = self.model["model"](signal)
        return (1 - self.dry) * estimate + self.dry * signal
    def save_to_file(self, audio_tensor, sr, audio_path, save_dir):        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        original_file_name = os.path.basename(audio_path)
        file_name_without_extension = os.path.splitext(original_file_name)[0]
        denoised_file_name = f"{file_name_without_extension}_denoised.wav"
        
        denoised_file_path = os.path.join(save_dir, denoised_file_name)
        
        torchaudio.save(denoised_file_path, audio_tensor, sample_rate=sr)
    
        return denoised_file_path

    def predict(self, audio_path, **kwargs) -> torch.Tensor:
        if not hasattr(self.model, "denoise_file"):
            raise NotImplementedError(f"{self.model_choice} doesn't have any supported methods")
        denoised_audio = self.model.denoise_file(audio_path)
        return denoised_audio.squeeze(-1) if denoised_audio.ndim == 3 else denoised_audio

    def __call__(self, audio_path: str = None, audio: torch.Tensor = None, save_to_file=False, offload=False, **kwargs) -> Any:
        assert exists(audio_path) or exists(audio), "Either audio_path or audio tensor is required"

        if save_to_file:
            audio_info = audio_ops.get_audio_info(audio_path)
            save_dir = kwargs.get("save_dir") or osp.join(settings.CACHE_DIR, "tmp", "denoiser")

        if isinstance(self.model, dict):
            denoised_audio = self.apply_denoiser(audio_path=audio_path, audio=audio, **kwargs)
        else:
            denoised_audio = self.predict(audio_path=audio_path, audio=audio, **kwargs)

        if save_to_file:
            denoised_audio = denoised_audio.detach().cpu()
            saved_file_path = self.save_to_file(denoised_audio, sr=audio_info.sample_rate,audio_path=audio_path, save_dir=save_dir)
            return saved_file_path

        elif offload:
            denoised_audio = denoised_audio.detach().cpu()

        return denoised_audio
