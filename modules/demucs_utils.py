# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from pathlib import Path
import subprocess

from dora.log import fatal
import torch as th
import torchaudio as ta

from demucs.apply import apply_model, BagOfModels
from demucs.audio import AudioFile, convert_audio, save_audio
from demucs.pretrained import get_model_from_args, add_model_flags, ModelLoadingError


def load_track(track, audio_channels, samplerate):
    errors = {}
    wav = None

    try:
        wav = AudioFile(track).read(
            streams=0, samplerate=samplerate, channels=audio_channels
        )
    except FileNotFoundError:
        errors["ffmpeg"] = "FFmpeg is not installed."
    except subprocess.CalledProcessError:
        errors["ffmpeg"] = "FFmpeg could not read the file."

    if wav is None:
        try:
            wav, sr = ta.load(str(track))
        except RuntimeError as err:
            errors["torchaudio"] = err.args[0]
        else:
            wav = convert_audio(wav, sr, samplerate, audio_channels)

    if wav is None:
        print(
            f"Could not load file {track}. " "Maybe it is not a supported file format? "
        )
        for backend, error in errors.items():
            print(
                f"When trying to load using {backend}, got the following error: {error}"
            )
        sys.exit(1)
    return wav


def get_parser():
    parser = argparse.ArgumentParser(
        "demucs.separate", description="Separate the sources for the given tracks"
    )
    parser.add_argument(
        "tracks", nargs="+", type=Path, default=[], help="Path to tracks"
    )
    add_model_flags(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("separated"),
        help="Folder where to put extracted tracks. A subfolder "
        "with the model name will be created.",
    )
    parser.add_argument(
        "--filename",
        default="{track}/{stem}.{ext}",
        help="Set the name of output file. \n"
        'Use "{track}", "{trackext}", "{stem}", "{ext}" to use '
        "variables of track name without extension, track extension, "
        "stem name and default output file extension. \n"
        'Default is "{track}/{stem}.{ext}".',
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda" if th.cuda.is_available() else "cpu",
        help="Device to use, default is cuda if available else cpu",
    )
    parser.add_argument(
        "--shifts",
        default=1,
        type=int,
        help="Number of random shifts for equivariant stabilization."
        "Increase separation time but improves quality for Demucs. 10 was used "
        "in the original paper.",
    )
    parser.add_argument(
        "--overlap", default=0.25, type=float, help="Overlap between the splits."
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--no-split",
        action="store_false",
        dest="split",
        default=True,
        help="Doesn't split audio in chunks. " "This can use large amounts of memory.",
    )
    split_group.add_argument(
        "--segment",
        type=int,
        help="Set split size of each chunk. "
        "This can help save memory of graphic card. ",
    )
    parser.add_argument(
        "--two-stems",
        dest="stem",
        metavar="STEM",
        help="Only separate audio into {STEM} and no_{STEM}. ",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--int24", action="store_true", help="Save wav output as 24 bits wav."
    )
    group.add_argument(
        "--float32", action="store_true", help="Save wav output as float32 (2x bigger)."
    )
    parser.add_argument(
        "--clip-mode",
        default="rescale",
        choices=["rescale", "clamp"],
        help="Strategy for avoiding clipping: rescaling entire signal "
        "if necessary  (rescale) or hard clipping (clamp).",
    )
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        "--flac", action="store_true", help="Convert the output wavs to flac."
    )
    format_group.add_argument(
        "--mp3", action="store_true", help="Convert the output wavs to mp3."
    )
    parser.add_argument(
        "--mp3-bitrate", default=320, type=int, help="Bitrate of converted mp3."
    )
    parser.add_argument(
        "--mp3-preset",
        choices=range(2, 8),
        type=int,
        default=2,
        help="Encoder preset of MP3, 2 for highest quality, 7 for "
        "fastest speed. Default is 2",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default=0,
        type=int,
        help="Number of jobs. This can increase memory usage but will "
        "be much faster when multiple cores are available.",
    )

    return parser


def load_demucs_model(args):
    try:
        model = get_model_from_args(args)
    except ModelLoadingError as error:
        fatal(error.args[0])

    if isinstance(model, BagOfModels):
        print(
            f"Selected model is a bag of {len(model.models)} models. "
            "You will see that many progress bars per track."
        )

    model.cpu()
    model.eval()
    return model


def main(opts=None):
    parser = get_parser()
    args = parser.parse_args(opts)

    model = load_demucs_model(args)

    if args.stem is not None and args.stem not in model.sources:
        fatal(
            'error: stem "{stem}" is not in selected model. STEM must be one of {sources}.'.format(
                stem=args.stem, sources=", ".join(model.sources)
            )
        )
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Separated tracks will be stored in {out.resolve()}")
    for track in args.tracks:
        if not track.exists():
            print(
                f"File {track} does not exist. If the path contains spaces, "
                'please try again after surrounding the entire path with quotes "".',
                file=sys.stderr,
            )
            continue
        print(f"Separating track {track}")
        wav = load_track(track, model.audio_channels, model.samplerate)

        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()
        sources = apply_model(
            model,
            wav[None],
            device=args.device,
            shifts=args.shifts,
            split=args.split,
            overlap=args.overlap,
            progress=True,
            num_workers=args.jobs,
            segment=args.segment,
        )[0]
        sources *= ref.std()
        sources += ref.mean()

        if args.mp3:
            ext = "mp3"
        elif args.flac:
            ext = "flac"
        else:
            ext = "wav"
        kwargs = {
            "samplerate": model.samplerate,
            "bitrate": args.mp3_bitrate,
            "preset": args.mp3_preset,
            "clip": args.clip_mode,
            "as_float": args.float32,
            "bits_per_sample": 24 if args.int24 else 16,
        }
        if args.stem is None:
            for source, name in zip(sources, model.sources):
                stem = out / args.filename.format(
                    track=track.name.rsplit(".", 1)[0],
                    trackext=track.name.rsplit(".", 1)[-1],
                    stem=name,
                    ext=ext,
                )
                stem.parent.mkdir(parents=True, exist_ok=True)
                save_audio(source, str(stem), **kwargs)
        else:
            sources = list(sources)
            stem = out / args.filename.format(
                track=track.name.rsplit(".", 1)[0],
                trackext=track.name.rsplit(".", 1)[-1],
                stem=args.stem,
                ext=ext,
            )
            stem.parent.mkdir(parents=True, exist_ok=True)
            save_audio(sources.pop(model.sources.index(args.stem)), str(stem), **kwargs)
            # Warning : after poping the stem, selected stem is no longer in the list 'sources'
            other_stem = th.zeros_like(sources[0])
            for i in sources:
                other_stem += i
            stem = out / args.filename.format(
                track=track.name.rsplit(".", 1)[0],
                trackext=track.name.rsplit(".", 1)[-1],
                stem="no_" + args.stem,
                ext=ext,
            )
            stem.parent.mkdir(parents=True, exist_ok=True)
            save_audio(other_stem, str(stem), **kwargs)
