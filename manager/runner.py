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


logger = get_logger("module_log")


class Runner:
    ALLOWED_PROCESSORS = [
        "downloader",
        "speaker_diarization",
        "voice_activity_detection",
        "music_separation",
        "denoise_audio",
        "gender_classification",
        "emotion_classification",
        "transcription",
        "superres",
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

        metadata = convert2wav(file_metadata["video"])

        dag_name = "voice_activity_detection"
        logger.info(f"Running pipeline -> {dag_name}")
        now = time.time()
        vad = self.run_dag(
            dag_name,
            audio_path=metadata["audio"],
            save_to_file=True,
            save_dir=osp.join(cache_dir, dag_name),
        )
        metadata[dag_name] = vad
        metadata[f"{dag_name}_proc_time"] = time.time() - now
        self.cleanup_dag(dag_name)

        dag_name = "denoise_audio"
        logger.info(f"Running pipeline -> {dag_name}")
        total_time = 0
        for v, va in tqdm(
            enumerate(metadata["voice_activity_detection"]["voice_activity"]),
            desc=dag_name,
        ):
            now = time.time()
            enhanced_audio = self.run_dag(
                dag_name,
                audio_path=va["filepath"],
                save_to_file=True,
                save_dir=osp.join(cache_dir, dag_name, osp.split(va["filepath"])[-1]),
            )
            proc_time = time.time() - now
            metadata["voice_activity_detection"]["voice_activity"][v].update(
                {dag_name: enhanced_audio, "enhancement_proc_time": proc_time}
            )
            total_time += proc_time
        metadata[f"{dag_name}_proc_time"] = total_time
        self.cleanup_dag(dag_name)
        return metadata


class ASR2TTSRunner(Runner):
    def __init__(self, configs, lazy_load=True) -> None:
        super().__init__(configs, lazy_load)

    def __call__(self, file_metadata, **kwargs):
        cache_dir = osp.join(settings.CACHE_DIR, "tmp", str(uuid4()))

        dag_name = "downloader"
        metadata, status = self.run_dag(
            dag_name, metadata=file_metadata, save_dir=cache_dir
        )
        self.cleanup_dag(dag_name)

        metadata["success"] = False
        if not status:
            logger.erro(f"Error downloading file: {metadata}")
            return metadata

        metadata["audio"] = convert2wav(metadata["audio"])

        dag_name = "voice_activity_detection"
        logger.info(f"Running pipeline -> {dag_name}")
        now = time.time()
        vad = self.run_dag(
            dag_name,
            audio_path=metadata["audio"],
            save_to_file=True,
            save_dir=osp.join(cache_dir, dag_name),
        )
        metadata[dag_name] = vad
        metadata[f"{dag_name}_proc_time"] = time.time() - now
        self.cleanup_dag(dag_name)

        dag_name = "enhancement"
        logger.info(f"Running pipeline -> {dag_name}")
        total_time = 0
        for v, va in tqdm(
            enumerate(metadata["voice_activity_detection"]["voice_activity"]),
            desc=dag_name,
        ):
            now = time.time()
            enhanced_audio = self.run_dag(
                dag_name,
                audio_path=va["filepath"],
                save_to_file=True,
                save_dir=osp.join(cache_dir, dag_name, osp.split(va["filepath"])[-1]),
            )
            proc_time = time.time() - now
            metadata["voice_activity_detection"]["voice_activity"][v].update(
                {dag_name: enhanced_audio, "enhancement_proc_time": proc_time}
            )
            total_time += proc_time
        metadata[f"{dag_name}_proc_time"] = total_time
        self.cleanup_dag(dag_name)

        filename = metadata["file"].split("/")[-1]
        ext = filename.split(".")[-1]
        save_dir = osp.join(cache_dir, "final", filename.replace(".", "_"))
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        chunk_metadata = {}
        dag_name = "superres"
        total_time = 0
        for v, va in tqdm(
            enumerate(metadata["voice_activity_detection"]["voice_activity"]),
            desc=dag_name,
        ):
            now = time.time()
            enhanced_audio = self.run_dag(
                dag_name,
                audio_path=va["enhancement"],
                save_dir=osp.join(cache_dir, dag_name, osp.split(va["filepath"])[-1]),
            )
            proc_time = time.time() - now
            metadata["voice_activity_detection"]["voice_activity"][v].update(
                {dag_name: enhanced_audio, "superres_proc_time": proc_time}
            )
            total_time += proc_time
            shutil.copy(va[dag_name], osp.join(save_dir, f"chunk_{v}.wav"))
            chunk_metadata[f"chunk_{v}.wav"] = {
                key: val
                for key, val in va.items()
                if key in ["start", "end", "duration"]
            }
        metadata[f"{dag_name}_proc_time"] = total_time
        self.cleanup_dag(dag_name)

        metadata["success"] = True
        with open(osp.join(save_dir, "graph.json"), "w") as file:
            json.dump(metadata, file)
        with open(osp.join(save_dir, "metadata.json"), "w") as file:
            json.dump(chunk_metadata, file)
        shutil.rmtree(cache_dir)
        return metadata
