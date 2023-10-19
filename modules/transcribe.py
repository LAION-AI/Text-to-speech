import torch
import whisper
from os import path as osp
from functools import partial
from .lang_list import LANGUAGE_NAME_TO_CODE
from .common import DSPBase
from . import audio as audio_ops
from utils.helpers import exists
from config import settings

cache_dir = osp.join(settings.CACHE_DIR, "weights", "transcription")


class TranscribeAudio(DSPBase):
    MODEL_CHOICES = {
        "openai_whisper_base": partial(
            whisper.load_model,
            name="base",
            download_root=osp.join(cache_dir, "openai-whisper-base"),
        ),
        "openai_whisper_medium": partial(
            whisper.load_model,
            name="medium",
            download_root=osp.join(cache_dir, "openai-whisper-medium"),
        ),
        "openai_whisper_large": partial(
            whisper.load_model,
            name="large",
            download_root=osp.join(cache_dir, "openai-whisper-large"),
        ),
    }
    def predict(
        self, audio_path: str = None, audio: torch.Tensor = None, **kwargs
    ) -> str:
        if exists(audio_path):
            if isinstance(self.model, whisper.Whisper):
                transcription = self.model.transcribe(audio_path)["text"]
            else:
                raise NotImplementedError(
                    f"{self.model_choice} doesn't have any supported methods"
                )
        else:
            transcription = self.model.transcribe(audio)["text"]
        return transcription
