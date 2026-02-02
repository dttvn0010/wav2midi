import sys
import librosa
import tensorflow as tf
import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple, Optional, Iterable, cast
from midi_utils import (
    model_output_to_notes,
    AUDIO_SAMPLE_RATE,
    FFT_HOP,
    AUDIO_N_SAMPLES,
    ANNOTATIONS_FPS
)

class Model:
    def __init__(self, model_path: str):
        self.model = tf.saved_model.load(str(model_path))

    def predict(self, x: npt.NDArray[np.float32]) -> Dict[str, npt.NDArray[np.float32]]:
        return {k: v.numpy() for k, v in cast(tf.keras.Model, self.model(x)).items()}


def window_audio_file(
    audio_original: npt.NDArray[np.float32], hop_size: int
) -> Iterable[Tuple[npt.NDArray[np.float32], Dict[str, float]]]:
    for i in range(0, audio_original.shape[0], hop_size):
        window = audio_original[i : i + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(
                window,
                pad_width=[[0, AUDIO_N_SAMPLES - len(window)]],
            )
        t_start = float(i) / AUDIO_SAMPLE_RATE
        window_time = {
            "start": t_start,
            "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
        }
        yield np.expand_dims(window, axis=-1), window_time

def get_audio_input(
    audio_path: str, overlap_len: int, hop_size: int
) -> Iterable[Tuple[npt.NDArray[np.float32], Dict[str, float], int]]:
    assert overlap_len % 2 == 0, f"overlap_length must be even, got {overlap_len}"

    audio_original, _ = librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)

    original_length = audio_original.shape[0]
    audio_original = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), audio_original])
    for window, window_time in window_audio_file(audio_original, hop_size):
        yield np.expand_dims(window, axis=0), window_time, original_length


def unwrap_output(
    output: npt.NDArray[np.float32],
    audio_original_length: int,
    n_overlapping_frames: int,
) -> np.array:
    if len(output.shape) != 3:
        return None

    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0:
        # remove half of the overlapping frames from beginning and end
        output = output[:, n_olap:-n_olap, :]

    output_shape = output.shape
    n_output_frames_original = int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    unwrapped_output = output.reshape(output_shape[0] * output_shape[1], output_shape[2])
    return unwrapped_output[:n_output_frames_original, :]  # trim to original audio length
 
def run_inference(
    audio_path: str,
    model_path: str,
) -> Dict[str, np.array]:

    model = Model(model_path)

    # overlap 30 frames
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    output = {"note": [], "onset": [], "contour": []}
    for audio_windowed, _, audio_original_length in get_audio_input(audio_path, overlap_len, hop_size):
        for k, v in model.predict(audio_windowed).items():
            output[k].append(v)

    unwrapped_output = {
        k: unwrap_output(np.concatenate(output[k]), audio_original_length, n_overlapping_frames) for k in output
    }

    return unwrapped_output

def convert_wav_to_midi(
    audio_path: str,
    model_path: str,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 127.70,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
    multiple_pitch_bends: bool = False,
    melodia_trick: bool = True,
    midi_tempo: float = 120,
) -> None:
    model_output = run_inference(audio_path, model_path)
    min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))

    midi_data, _ = model_output_to_notes(
        model_output,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_len,  # convert to frames
        min_freq=minimum_frequency,
        max_freq=maximum_frequency,
        multiple_pitch_bends=multiple_pitch_bends,
        melodia_trick=melodia_trick,
        midi_tempo=midi_tempo,
    )

    midi_path = audio_path.replace(".wav", ".mid")
    midi_data.write(str(midi_path))

audio_path = sys.argv[1]
model_path = "model"

convert_wav_to_midi(
    audio_path,
    model_path
)

