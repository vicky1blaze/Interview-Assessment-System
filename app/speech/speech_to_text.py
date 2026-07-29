import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

# Configuration

SAMPLE_RATE = 16000   
CHANNELS = 1              
MODEL_NAME = "base"

model = WhisperModel(
    MODEL_NAME,
    device="cpu",
    compute_type="int8"
)

def record(audio_path):
    print("\nPress Enter to start recording")
    input()

    print("Recording...")
    print("Press Enter to stop recording")

    recording = []

    def callback(indata, frames, time, status):
        recording.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=callback
    ):
        input()

    audio = b""

    audio = np.concatenate(recording, axis=0)

    sf.write(audio_path, audio, SAMPLE_RATE)

    print("Recording saved!\n")

def speech_to_text(audio_path, candidate_data, candidate_id, question_id):
    segments, info = model.transcribe(audio_path)

    if question_id not in candidate_data[candidate_id]["speech"]["answers"]:
         candidate_data[candidate_id]["speech"]["answers"][question_id] = ""

    for segment in segments:
            candidate_data[candidate_id]["speech"]["answers"][question_id] += segment.text
            candidate_data[candidate_id]["speech"]["combined_answer"] += segment.text

    return candidate_data
