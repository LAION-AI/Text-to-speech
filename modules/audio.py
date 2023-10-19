import torchaudio
import lameenc
from pathlib import Path
from os import system as os_system

FFMPEG_BIN = "ffmpeg"

def load_audio(audio_path):
    audio, sr = torchaudio.load(audio_path)
    return audio, sr

def normalize_audio(wav):
    return wav / max(wav.abs().max().item(), 1)

def convert_channels(wav, channels):
    if wav.shape[0] != channels:
        wav = wav.mean(dim=0, keepdim=True).expand(channels, -1)
    return wav

def convert_audio(wav, from_sr, to_sr, channels):
    if from_sr != to_sr:
        wav = torchaudio.transforms.Resample(from_sr, to_sr)(wav)
    return convert_channels(wav, channels)

def save_audio(wav, path, sr, bitrate=320, bits_per_sample=16):
    path = Path(path)
    if path.suffix.lower() == ".mp3":
        encode_mp3(wav, path, sr, bitrate)
    else:
        torchaudio.save(str(path), wav, sample_rate=sr, bits_per_sample=bits_per_sample)

def encode_mp3(wav, path, sr, bitrate):
    wav = (wav.clamp_(-1, 1) * (2 ** 15 - 1)).short().data.cpu().numpy().T
    mp3_data = lameenc.Encoder().set_bit_rate(bitrate).set_in_sample_rate(sr).set_channels(1).encode(wav.tobytes())
    with open(path, "wb") as f:
        f.write(mp3_data)

def convert_and_trim_ffmpeg(src_file, dst_file, sr, start_tm, end_tm):
    cmd = f'{FFMPEG_BIN} -i {src_file} -ar {sr} -ac 1 -ss {start_tm} -to {end_tm} {dst_file} -y -loglevel panic'
    os_system(cmd)

def get_duration(wave_file, sr=16000):
    y, _ = torchaudio.load(wave_file)
    return len(y[0]) / sr

def convert2wav(audio_path):
    ext = Path(audio_path).suffix.lower()
    supported_ext = [".sph", ".wav", ".mp3", ".flac", ".ogg", ".mp4"]

    if ext not in supported_ext:
        raise NotImplementedError(f"Audio format {ext} is not supported")

    if ext != ".wav":
        dst_path = audio_path.replace(ext, ".wav")
        cmd = f'{FFMPEG_BIN} -i {audio_path} {dst_path}'
        os_system(cmd)
        return dst_path
    else:
        return audio_path

# Usage example
if __name__ == '__main__':
    audio, sr = load_audio("audio_path")
    audio = normalize_audio(audio)
    audio = convert_audio(audio, sr, 44100, 1)
    save_audio(audio, "output_path", 44100)
