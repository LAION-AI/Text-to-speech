from functools import partial
from os import path as osp, remove as os_remove
from pathlib import Path
from pyannote.audio import Pipeline
from webrtcvad import Vad

from .common import Base
from . import audio as audio_ops
from .vad_utils import vad_audio_segment
from config import settings

cache_dir = osp.join(settings.CACHE_DIR, "weights", "vad")


class VoiceActivityDetection(Base):
    MODEL_CHOICES = {
        "pyannote_voice_activity_detection": {
            "model": partial(
                Pipeline.from_pretrained,
                checkpoint_path="pyannote/voice-activity-detection",
                use_auth_token=settings.huggingface.HF_TOKEN,
                cache_dir=osp.join(cache_dir, "pyannote-voice-activity-detection"),
            ),
            "target": "vad_with_pyannote",
        },
        "webrtc_voice_activity_detection": {
            "model": partial(Vad, mode=3),
            "target": "vad_with_webrtc",
        },
    }

    def vad_with_pyannote(self, audio_path, save_to_file=False, **kwargs):
        audio_info = audio_ops.get_audio_info(audio_path)
        output = self.model["model"](audio_path)

        if save_to_file:
            audio, _ = audio_ops.load_audio(audio_path)
            save_dir = kwargs.get("save_dir") or osp.join(
                settings.CACHE_DIR,
                "tmp",
                "vad",
                f"{osp.splitext(osp.split(audio_path)[-1])[0]}",
            )

        va_list = []
        va_duration = 0
        for speech in output.get_timeline().support():
            meta = {
                "start": round(speech.start, 1),
                "stop": round(speech.end, 1),
                "duration": None,
                "filepath": None,
            }
            meta["duration"] = meta["stop"] - meta["start"]
            va_duration += meta["duration"]

            if save_to_file:
                meta["filepath"] = self.save_to_file(
                    audio=audio,
                    sr=audio_info.sample_rate,
                    save_dir=save_dir,
                    start_dur=meta["start"],
                    stop_dur=meta["stop"],
                )

            va_list.append(meta)
        return {
            "voice_activity": va_list,
            "va_duration": va_duration,
            "total_duration": audio_info.length,
        }

    def vad_with_webrtc(
        self,
        audio_path,
        min_silence=0.1,
        min_duration=1.0,
        max_duration=20.0,
        save_to_file=False,
        **kwargs,
    ):
        print("Reading from {}".format(audio_path))
        audio_info = audio_ops.get_audio_info(audio_path)
        dur = audio_info.length
        print("Audio duration is {}s".format(dur))

        convert_file = audio_path[:-4] + "_convert.wav"
        if not audio_ops.convert2wav_ffmpeg(
            audio_path, convert_file, sr=audio_info.sample_rate
        ):
            return

        segments = vad_audio_segment(self.model["model"], convert_file)
        if len(segments) == 0:
            print("No segments to split")
            return

        try:
            os_remove(convert_file)
        except:
            pass

        # make a segment using min/max length
        final_list = []
        temp_segment = segments[0]
        for x in segments[1:]:
            cur_duration = x[1] - x[0]
            temp_duration = temp_segment[1] - temp_segment[0]

            # try to split on silences no shorter than min duration
            if x[0] - temp_segment[1] <= min_silence:
                temp_segment[1] = x[1]
                continue

            if cur_duration + temp_duration > max_duration:
                final_list.append(temp_segment)
                temp_segment = x
            else:
                temp_segment[1] = x[1]
        final_list.append(temp_segment)

        if final_list[-1][1] - final_list[-1][0] <= min_duration:
            mean_time = (final_list[-2][0] + final_list[-1][1]) / 2
            final_list[-2][1] = mean_time
            final_list[-1][0] = mean_time

        # split audio
        split_dir = audio_path[:-4] + "_split"
        Path(split_dir).mkdir(exist_ok=True, parents=True)
        final_list[0][0] = 0
        final_list[-1][1] = dur
        print("Chunks will be saved to {}".format(split_dir))

        va_list = []
        va_duration = 0
        for i, seg in enumerate(final_list):
            stime, etime = seg
            if i == 0:
                etime = (etime + final_list[i + 1][0]) / 2
            elif i == len(final_list) - 1:
                stime = (stime + final_list[i - 1][1]) / 2
            else:
                stime = (stime + final_list[i - 1][1]) / 2
                etime = (etime + final_list[i + 1][0]) / 2

            meta = {
                "start": round(stime, 1),
                "stop": round(etime, 1),
                "duration": None,
                "filepath": None,
            }
            meta["duration"] = meta["stop"] - meta["start"]
            va_duration += meta["duration"]

            if save_to_file:
                split_audio_path = "{}_{}.wav".format(
                    osp.basename(audio_path)[:-4], i + 1
                )
                split_audio_path = osp.join(split_dir, split_audio_path)
                audio_ops.trim_audio_ffmpeg(
                    audio_path, stime, etime, split_audio_path, audio_info.sample_rate
                )
                meta["filepath"] = split_audio_path

            va_list.append(meta)

        return {
            "voice_activity": va_list,
            "va_duration": va_duration,
            "total_duration": audio_info.length,
        }
