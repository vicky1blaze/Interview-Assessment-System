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

def speech_to_text(audio_path, candidates_data, corpus_data, candidate_id, question_id):
    """
    Transcribe audio and store in the new hierarchy:
    - candidates_data[candidate_id]["responses"][question_id]["speech"]["transcript"]
    - corpus_data[candidate_id]["responses"][question_id]["speech"]["transcript"]
    
    Also stores audio_path in candidates.
    """
    segments, info = model.transcribe(audio_path)

    transcript = ""
    for segment in segments:
        transcript += segment.text

    # Store in candidates.json
    if "responses" not in candidates_data["candidates"][candidate_id]:
        candidates_data["candidates"][candidate_id]["responses"] = {}
    
    if question_id not in candidates_data["candidates"][candidate_id]["responses"]:
        candidates_data["candidates"][candidate_id]["responses"][question_id] = {
            "text": {"answer": ""},
            "speech": {"audio_path": None, "transcript": None},
            "video": {"video_path": None}
        }
    
    candidates_data["candidates"][candidate_id]["responses"][question_id]["speech"]["audio_path"] = audio_path
    candidates_data["candidates"][candidate_id]["responses"][question_id]["speech"]["transcript"] = transcript

    # Store in corpus.json
    if "responses" not in corpus_data["corpus"][candidate_id]:
        corpus_data["corpus"][candidate_id]["responses"] = {}
    
    if question_id not in corpus_data["corpus"][candidate_id]["responses"]:
        corpus_data["corpus"][candidate_id]["responses"][question_id] = {
            "text": {"processed": {}, "features": {}},
            "speech": {"transcript": "", "features": {}},
            "video": {"features": {}},
            "evaluation": {}
        }

    corpus_data["corpus"][candidate_id]["responses"][question_id]["speech"]["transcript"] = transcript

    return candidates_data, corpus_data
