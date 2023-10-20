from functools import partial
from os import path as osp
import os
import torchaudio
from .common import Base
from . import audio
from config import settings
from pydub import AudioSegment
from pydub.silence import split_on_silence
from typing import Any

class AudioChunking(Base):
    MODEL_CHOICES = {
        "pydub_chunking": {
            "target": "chunk_by_silence",
        }
    }
    def __init__(self, model_choice: str, **kwargs) -> None:
        super().__init__(model_choice, **kwargs)

    def chunk_by_silence(self, audio_path, silence_len=500, silence_thresh=-30, **kwargs) -> Any:
        audio_info = audio.get_audio_info(audio_path)
        audio_segment = AudioSegment.from_wav(audio_path)

        # Use pydub to split audio based on silence
        chunks = split_on_silence(audio_segment, 
                                  min_silence_len=silence_len,
                                  silence_thresh=silence_thresh)

        chunk_list = []
        total_chunk_duration = 0

        for i, chunk in enumerate(chunks):
            meta = {
                "duration": chunk.duration_seconds,
                "filepath": None,
                "sample_rate": chunk.frame_rate,
            }

            total_chunk_duration += meta["duration"]

            if "save_to_file" in kwargs and kwargs["save_to_file"]:
                meta["filepath"] = self.save_to_file(chunk, chunk.frame_rate, audio_path, i, save_dir=kwargs["save_dir"])

            chunk_list.append(meta)

        return {
            "audio_chunks": chunk_list,
            "total_chunk_duration": total_chunk_duration,
            "total_audio_duration": audio_info.length,
        }

    def save_to_file(self, audio_chunk, sr, audio_path, chunk_idx, save_dir):       
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        original_file_name = osp.basename(audio_path)
        file_name_without_extension = osp.splitext(original_file_name)[0]
        chunk_file_name = f"{file_name_without_extension}_chunk_{chunk_idx}.wav"
        chunk_file_path = osp.join(save_dir, chunk_file_name)

        audio_chunk.export(chunk_file_path, format="wav")
        
        return chunk_file_path

    def __call__(self, audio_path: str = None, **kwargs) -> Any:
        assert audio_path is not None, "audio_path is required"

        audio_chunks_info = self.chunk_by_silence(audio_path, **kwargs)

        return audio_chunks_info
