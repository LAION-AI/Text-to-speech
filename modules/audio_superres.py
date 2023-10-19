from os import path as osp
from pathlib import Path
from voicefixer import VoiceFixer

from .common import DSPBase
from config import settings

cache_dir = osp.join(settings.CACHE_DIR, "weights", "enhancement")


class SuperResAudio(DSPBase):
    MODEL_CHOICES = {
        "voicefixer": {
            "model": VoiceFixer,
            "target": "sr_with_voicefixer",
        },
    }

    def sr_with_voicefixer(self, audio_path, **kwargs):
        save_dir = kwargs.get("save_dir") or osp.join(
            settings.CACHE_DIR, "tmp", "sr", osp.split(audio_path)[-1]
        )
        Path(osp.dirname(save_dir)).mkdir(exist_ok=True, parents=True)
        self.model["model"].restore(
            input=audio_path,  # low quality .wav/.flac file
            output=save_dir,  # save file path
            cuda=False,  # GPU acceleration
            mode=0,
        )
        return save_dir
