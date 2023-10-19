import torch
from os import path as osp
from functools import partial
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from speechbrain.pretrained.interfaces import foreign_class

from .common import Base
from . import audio as audio_ops
from utils.helpers import exists
from config import settings

cache_dir = osp.join(settings.CACHE_DIR, "weights", "classification")


class ClassifySpeakerGender(Base):
    MODEL_CHOICES = {
        "transformers_wav2vec2_xlsr_300m": {
            "processor": partial(
                AutoFeatureExtractor.from_pretrained,
                pretrained_model_name_or_path="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
                cache_dir=osp.join(
                    cache_dir, "wav2vec2-large-xlsr-53-gender-recognition-librispeech"
                ),
            ),
            "classifier": partial(
                AutoModelForAudioClassification.from_pretrained,
                pretrained_model_name_or_path="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
                cache_dir=osp.join(
                    cache_dir, "wav2vec2-large-xlsr-53-gender-recognition-librispeech"
                ),
                num_labels=2,
                label2id={"female": 0, "male": 1},
                id2label={0: "female", 1: "male"},
            ),
            "target": "classify_wav2vec2_xlsr",
        }
    }

    def classify_wav2vec2_xlsr(
        self,
        audio_path: str = None,
        audio: torch.Tensor = None,
        sr: int = None,
        **kwargs
    ):
        if not exists(audio):
            audio, sr = audio_ops.load_audio(audio_path)
        assert exists(sr), "Sampling rate is required"

        # Transform to Mono
        audio = audio_ops.convert_audio(
            audio, from_samplerate=sr, to_samplerate=self.sampling_rate, channels=1
        )
        sr = self.sampling_rate

        audio = audio_ops.trim_audio(audio, self.max_audio_len, sr)

        audio = audio.squeeze().numpy()
        input_tensor = self.model["processor"](
            audio, sampling_rate=sr, return_tensors="pt"
        )

        with torch.no_grad():
            logits = self.model["classifier"](**input_tensor).logits

        predicted_class_ids = torch.argmax(logits).item()
        predicted_label = self.model["classifier"].config.id2label[predicted_class_ids]
        return predicted_label


class classifySpeakerEmotion(Base):
    MODEL_CHOICES = {
        "speechbrain_wav2vec2_iemocap": partial(
            foreign_class,
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=osp.join(
                cache_dir,
                "speechbrain-emotion-recognition-wav2vec2-IEMOCAP",
            ),
        )
    }

    def predict(self, audio_path: str = None, **kwargs) -> str:
        _, _, _, label = self.model.classify_file(audio_path)
        return label[0] if isinstance(label, list) else label
