from functools import lru_cache
from os import path as osp
from uuid import uuid4
from tqdm import tqdm
from pathlib import Path
import inspect
import shutil
import json
import time

from modules.audio import convert2wav
from config import settings
from utils.io import load_configs, merge_configs
from utils.helpers import exists, get_obj_from_str
from utils.loggers import get_logger
import torchaudio

logger = get_logger("module_log")


class Runner:
    ALLOWED_PROCESSORS = [
        "downloader",
        "speaker_diarization",
        "chunking",
        "music_separation",
        "denoise_audio",
        "gender_classification",
        "emotion_classification",
        "transcription",
        "audio_superres",
    ]

    def __init__(self, configs, lazy_load=True) -> None:
        self.config = load_configs(configs)
        self.lazy_load = lazy_load

        self.processors = {}
        if not hasattr(self.config, "processors"):
            self.config = merge_configs(self.config, {"processors": []})
        for proc in self.config.processors:
            assert (
                proc.name in self.ALLOWED_PROCESSORS
            ), f"Processor {proc.name} is not supported"
            assert (
                proc.name not in self.processors
            ), f"A processor already exists with the name {proc.name}"

            obj = get_obj_from_str(proc.target)
            if inspect.isclass(obj) and not self.lazy_load:
                self.processors[proc.name] = {"obj": obj(**proc.args), "loaded": True}
            else:
                self.processors[proc.name] = {"obj": obj, "loaded": False}

    def load_processors(self, names=None, reload=False):
        if not (self.lazy_load or reload):
            return
        if exists(names) and isinstance(names, str):
            names = [names]

        for proc in self.config.processors:
            if (
                (not exists(names) or proc.name in names)
                and (not self.processors[proc.name]["loaded"] or reload)
                and inspect.isclass(self.processors[proc.name]["obj"])
            ):
                if hasattr(proc, "args"):
                    self.processors[proc.name]["obj"] = self.processors[proc.name][
                        "obj"
                    ](**proc.args)
                else:
                    self.processors[proc.name]["obj"] = self.processors[proc.name][
                        "obj"
                    ]()
                self.processors[proc.name]["loaded"] = True

    def offload_processors(self, names=None):
        if exists(names) and isinstance(names, str):
            names = [names]

        for proc in self.config.processors:
            if not exists(names) or proc.name in names:
                del self.processors[proc.name]
                self.processors[proc.name] = {
                    "obj": get_obj_from_str(proc.target),
                    "loaded": False,
                }
                logger.info(f"Dag {proc.name} deleted!!!")

    def resolve_dag_processor(self, name):
        raw_name = name
        name = name.split(".")[0]
        assert name in self.processors, f"Processor for dag {raw_name} is not declared"
        self.load_processors(name)
        return self.processors[name]["obj"]

    @lru_cache()
    def refactor_dag_name_with_ordinal(self, name):
        name_splits = name.split(".")

        if name_splits[0] not in self.processed_dags:
            self.processed_dags[name_splits[0]] = 0
        else:
            self.processed_dags[name_splits[0]] += 1

        if len(name_splits) == 1:
            name = name + f".{len(self.processed_dags[name])}"
        else:
            name = name

    def run_dag(self, name, **kwargs):
        processor = self.resolve_dag_processor(name)
        return processor(**kwargs)

    def cleanup_dag(self, name):
        if self.lazy_load:
            logger.info("Cleaning up dag: %s", name)
            self.offload_processors(name)


class YoutubeRunner(Runner):
    def __init__(self, configs, lazy_load=True) -> None:
        super().__init__(configs, lazy_load)

    def __call__(self, file_metadata, **kwargs):
        cache_dir = osp.join(settings.CACHE_DIR, "tmp", str(uuid4()))
        # print(file_metadata["video"].split("/")[-1])
        wav_path = convert2wav(file_metadata["video"])
        dag_name = "chunking"
        logger.info(f"Running pipeline -> {dag_name}")
        now = time.time()
        audio_chunks = self.run_dag(
            dag_name,
            audio_path=wav_path,
            save_to_file=True,
            save_dir=osp.join("data", file_metadata["video"].split("/")[-1][:-4],"chunked_audio"),
        )
        file_metadata[dag_name] = audio_chunks
        file_metadata[f"{dag_name}_proc_time"] = time.time() - now
        dag_name = "denoise_audio"
        logger.info(f"Running pipeline -> {dag_name}")
        total_time = 0
        for v, va in tqdm(
            enumerate(file_metadata["chunking"]["audio_chunks"]),
            desc=dag_name,
        ):
            now = time.time()
            enhanced_audio = self.run_dag(
                dag_name,
                audio_path=va["filepath"],
                save_to_file=True,
                save_dir=osp.join("data", file_metadata["video"].split("/")[-1][:-4],"denoise_audio"),
            )
            proc_time = time.time() - now
            file_metadata["chunking"]["audio_chunks"][v].update(
                {dag_name: enhanced_audio, "enhancement_proc_time": proc_time}
            )
            total_time += proc_time
        file_metadata[f"{dag_name}_proc_time"] = total_time
        self.cleanup_dag(dag_name)

        dag_name = "audio_superres"
        logger.info(f"Running pipeline -> {dag_name}")
        total_time = 0

        for v, va in tqdm(
            enumerate(file_metadata["chunking"]["audio_chunks"]),
            desc=dag_name,
        ):
            now = time.time()
            enhanced_audio = self.run_dag(
                dag_name,
                audio_path=va["denoise_audio"],
                save_to_file=True,
                save_dir=osp.join("data", file_metadata["video"].split("/")[-1][:-4],"superres_audio"),
            )
            proc_time = time.time() - now
            file_metadata["chunking"]["audio_chunks"][v].update(
                {dag_name: enhanced_audio, "enhancement_proc_time": proc_time}
            )
            total_time += proc_time
        file_metadata[f"{dag_name}_proc_time"] = total_time
        self.cleanup_dag(dag_name)

        dag_name = "transcription"
        logger.info(f"Running pipeline -> {dag_name}")
        total_time = 0
        print(file_metadata)
        for v, va in tqdm(
            enumerate(file_metadata["chunking"]["audio_chunks"]),
            desc=dag_name,
        ):
            now = time.time()
            transcription = self.run_dag(
                dag_name,
                audio_path=va["audio_superres"]
            )
            proc_time = time.time() - now
            total_time += proc_time
            transcript_path = osp.join("data", file_metadata["video"].split("/")[-1][:-4],"transcription")
            
            make_path = Path(transcript_path)
            make_path.mkdir(parents=True, exist_ok=True)

            file_path = transcript_path+"/"+va["filepath"].split("/")[-1][:-4]+".txt"
            with open(file_path, 'w', encoding='utf-8') as transcript_file:
                transcript_file.write(transcription)
            file_metadata["chunking"]["audio_chunks"][v].update(
                    {dag_name: enhanced_audio, "enhancement_proc_time": proc_time}
                )
        file_metadata[f"{dag_name}_proc_time"] = total_time
        self.cleanup_dag(dag_name)
        return file_metadata
