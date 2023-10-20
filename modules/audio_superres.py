from os import path as osp
import os
from pathlib import Path
from audiosr import super_resolution
from functools import partial
import argparse
from .common import Base
from modules.audio_superres_utils import load_audiosr
from voicefixer import VoiceFixer
from config import settings

cache_dir = osp.join(settings.CACHE_DIR, "weights", "enhancement")


class SuperResAudio(Base):
    MODEL_CHOICES = {
        "audiosr": {
            "model": partial(
                load_audiosr,
                args=argparse.Namespace(
                    **{
                        "model_name": None,
                        "device": "auto",
                    }
                )
            ),
            "target": "sr_with_voicefixer",
        },
        "voicefixer": {
            "model": VoiceFixer,
            "target": "sr_with_voicefixer",
        },
    }

    def sr_with_audiosr(self, audio_path):
        waveform = super_resolution(
            self.model,
            audio_path,
            guidance_scale=3.5,
            ddim_steps=50,
            latent_t_per_second=12.8
        )
        return waveform
    def sr_with_voicefixer(self, audio_path, **kwargs):
        save_dir = kwargs.get("save_dir")
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        original_file_name = osp.basename(audio_path)
        self.model["model"].restore(
            input=audio_path,  # low quality .wav/.flac file
            output=osp.join(save_dir,original_file_name),  # save file path
            cuda=True,  # GPU acceleration
            mode=0,
        )
        return save_dir
