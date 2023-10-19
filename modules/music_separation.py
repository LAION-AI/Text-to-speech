import torch
import argparse
import sys
from dora.log import fatal
from os import path as osp
from functools import partial
from demucs.apply import apply_model, BagOfModels
from demucs.htdemucs import HTDemucs

from .common import Base
from .demucs_utils import load_demucs_model, load_track
from . import audio as audio_ops
from utils.helpers import exists
from config import settings

cache_dir = osp.join(settings.CACHE_DIR, "weights", "partition")


class PartitionAudio(Base):
    MODEL_CHOICES = {
        "meta_demucs_htdemucs": {
            "model": partial(
                load_demucs_model,
                args=argparse.Namespace(**{"segment": None, "name": "htdemucs"}),
            ),
            "target": "partition_with_demucs",
        },
        "meta_demucs_htdemucs_ft": {
            "model": partial(
                load_demucs_model,
                args=argparse.Namespace(**{"segment": None, "name": "htdemucs_ft"}),
            ),
            "target": "partition_with_demucs",
        },
        "meta_demucs_htdemucs_6s": {
            "model": partial(
                load_demucs_model,
                args=argparse.Namespace(**{"segment": None, "name": "htdemucs_6s"}),
            ),
            "target": "partition_with_demucs",
        },
        "meta_demucs_htdemucs_mmi": {
            "model": partial(
                load_demucs_model,
                args=argparse.Namespace(**{"segment": None, "name": "htdemucs_mmi"}),
            ),
            "target": "partition_with_demucs",
        },
        "meta_demucs_mdx": {
            "model": partial(
                load_demucs_model,
                args=argparse.Namespace(**{"segment": None, "name": "mdx"}),
            ),
            "target": "partition_with_demucs",
        },
        "meta_demucs_mdx_q": {
            "model": partial(
                load_demucs_model,
                args=argparse.Namespace(**{"segment": None, "name": "mdx_q"}),
            ),
            "target": "partition_with_demucs",
        },
        "meta_demucs_mdx_extra": {
            "model": partial(
                load_demucs_model,
                args=argparse.Namespace(**{"segment": None, "name": "mdx_extra"}),
            ),
            "target": "partition_with_demucs",
        },
        "meta_demucs_mdx_extra_q": {
            "model": partial(
                load_demucs_model,
                args=argparse.Namespace(**{"segment": None, "name": "mdx_extra_q"}),
            ),
            "target": "partition_with_demucs",
        },
    }

    def partition_with_demucs(
        self, audio_path, save_to_file=False, save_partitions=None, **kwargs
    ):
        stem = "vocals"
        ext = kwargs.get("ext", "wav")
        float32 = False  # output as float 32 wavs, unsused if 'mp3' is True.
        int24 = False
        segment = kwargs.get("segment", 15)

        if exists(save_partitions) and not isinstance(save_partitions, int):
            save_partitions = [int(save_partitions)]
        if save_to_file:
            save_dir = kwargs.get("save_dir") or osp.join(
                settings.CACHE_DIR,
                "tmp",
                "partitions",
                f"{osp.splitext(osp.split(audio_path)[-1])[0]}",
            )

        max_allowed_segment = float("inf")
        if isinstance(self.model["model"], HTDemucs):
            max_allowed_segment = float(self.model["model"].segment)
        elif isinstance(self.model["model"], BagOfModels):
            max_allowed_segment = self.model["model"].max_allowed_segment
        if segment is not None and segment > max_allowed_segment:
            fatal(
                "Cannot use a Transformer model with a longer segment "
                f"than it was trained for. Maximum segment is: {max_allowed_segment}"
            )

        if stem is not None and stem not in self.model["model"].sources:
            fatal(
                'error: stem "{stem}" is not in selected model. STEM must be one of {sources}.'.format(
                    stem=stem, sources=", ".join(self.model["model"].sources)
                )
            )

        if not osp.isfile(audio_path):
            print(
                f"File {audio_path} does not exist. If the path contains spaces, "
                'please try again after surrounding the entire path with quotes "".',
                file=sys.stderr,
            )
            raise FileNotFoundError(audio_path)

        print(f"Separating track {audio_path}")
        wav = load_track(
            audio_path,
            self.model["model"].audio_channels,
            self.model["model"].samplerate,
        )

        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()
        sources = apply_model(
            self.model["model"],
            wav[None],
            device="cpu",
            shifts=kwargs.get("shifts", 1),
            split=kwargs.get("split", False),
            overlap=kwargs.get("overlap", 0.25),
            progress=True,
            num_workers=0,
            segment=segment,
        )[0]
        sources *= ref.std()
        sources += ref.mean()

        kwargs = {
            "samplerate": self.model["model"].samplerate,
            "bitrate": kwargs.get("mp3_bitrate"),
            "preset": kwargs.get("mp3_preset"),
            "clip": kwargs.get("clip_mode", "rescale"),
            "as_float": float32,
            "bits_per_sample": 24 if int24 else 16,
        }
        if stem is None:
            partitions = []
            for source, name in zip(sources, self.model["model"].sources):
                if save_to_file and (
                    not exists(save_partitions) or len(partitions) in save_partitions
                ):
                    save_path = self.save_to_file(
                        source,
                        save_dir=osp.join(save_dir, f"partition_{len(partitions)}.wav"),
                    )
                    partitions.append(save_path)
                else:
                    partitions.append(source)
        else:
            sources = list(sources)
            partitions = []
            for i in range(0, 2):
                if not i:
                    source = sources.pop(self.model["model"].sources.index(stem))
                else:
                    # Warning : after poping the stem, selected stem is no longer in the list 'sources'
                    source = torch.zeros_like(sources[0])
                    for src in sources:
                        source += src

                if save_to_file and (
                    not exists(save_partitions) or len(partitions) in save_partitions
                ):
                    save_path = self.save_to_file(
                        source,
                        save_dir=save_dir,
                    )
                    partitions.append(save_path)
                else:
                    partitions.append(source)

        return torch.vstack(partitions) if not save_to_file else partitions

    def predict(self, audio_path, **kwargs) -> torch.Tensor:
        if hasattr(self.model, "separate_file"):
            paritions = self.model.separate_file(audio_path)
        else:
            raise NotImplementedError(
                f"{self.model_choice} doesn't have any supported methods"
            )
        return paritions
