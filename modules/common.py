import torch
from uuid import uuid4

from typing import Union, Optional, Any
from functools import partial
import os
from abc import abstractmethod

from . import audio as audio_ops
from utils.helpers import exists
from utils.loggers import get_logger


logger = get_logger("module_log")


class Base:
    MODEL_CHOICES = {}

    def __init__(
        self,
        model_choice: str,
        sampling_rate: int = 16000,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        max_audio_len: int = 5,
        **kwargs,
    ) -> None:
        self.model_choice = model_choice.lower()
        assert (
            self.model_choice in self.MODEL_CHOICES
        ), f"Unrecognized model choice {self.model_choice}"
        model = self.MODEL_CHOICES[self.model_choice]
        if isinstance(model, dict):
            self.model = {}
            for key, value in model.items():
                if key in ["target"]:
                    continue
                self.model[key] = value(**kwargs)
        elif isinstance(model, partial):
            self.model = model(**kwargs)
        else:
            raise NotImplementedError("Not sure how to handle this model choice")

        self.sampling_rate = sampling_rate
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_audio_len = max_audio_len

        self.__post__init__()

    def __post__init__(self):
        for key, value in self.MODEL_CHOICES.items():
            if (
                isinstance(value, dict)
                and "target" in value
                and isinstance(value["target"], str)
            ):
                self.MODEL_CHOICES[key]["target"] = getattr(self, value["target"])

    @abstractmethod
    def predict(self, **kwargs):
        self.model(**kwargs)

    def __call__(
        self, audio_path: str = None, audio: torch.Tensor = None, **kwargs
    ) -> Any:
        assert exists(audio_path) or exists(
            audio
        ), "Either audio_path or audio tensor is required"
        if isinstance(self.model, dict):
            prediction = self.MODEL_CHOICES[self.model_choice]["target"](
                audio_path=audio_path, audio=audio, **kwargs
            )
        else:
            prediction = self.predict(audio_path=audio_path, audio=audio, **kwargs)
        return prediction

    def save_to_file(self, audio, sr, save_dir, start_dur=None, stop_dur=None):
        # Handling audio with more than 2 dimensions
        if audio.ndim > 2:
            print(f"Warning: Audio has {audio.ndim} dimensions, averaging over channels for simplicity.")
            audio = torch.mean(audio, dim=-1)

        if exists(start_dur):
            start_sample = round(start_dur * sr)
            audio = audio[start_sample:]
            
        if exists(stop_dur):
            stop_sample = round(stop_dur * sr)
            audio = audio[:stop_sample]

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        save_path = (
            os.path.join(save_dir, f"{str(uuid4())}.wav")
            if not os.path.splitext(save_dir)[-1]
            else save_dir
        )
        audio_ops.save_audio(wav=audio, path=save_path, sr=sr)
        return save_path
