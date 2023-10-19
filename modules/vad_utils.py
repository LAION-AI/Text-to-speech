import contextlib
import wave


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (16000, 22050, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_audio_segment(model, audio_file, gap_size=0.5, frame_duration=10):
    """
    :param audio_file:
    :param gap_size: gap between neighbour segments (seconds)
    :param frame_duration: frame step (mili seconds)
    :return:
    """
    audio, sample_rate = read_wave(audio_file)
    frames = frame_generator(frame_duration, audio, sample_rate)
    frames = list(frames)

    vad_segment = []
    for i, frame in enumerate(frames):
        is_speech = model.is_speech(frame.bytes, sample_rate)
        if is_speech:
            vad_segment.append([frame.timestamp, frame.timestamp + frame.duration])

    if len(vad_segment) == 0:
        return []
    audio_segment = [vad_segment[0]]
    for x in vad_segment[1:]:
        if x[0] <= audio_segment[-1][1] + gap_size:
            audio_segment[-1][1] = x[1]
        else:
            audio_segment.append(x)

    return audio_segment
