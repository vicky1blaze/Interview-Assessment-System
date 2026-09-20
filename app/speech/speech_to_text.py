"""
speech_to_text.py - Lazy-loaded Faster-Whisper Speech Recognition & Audio Capture.

Manages audio recording via sounddevice and lazy-loaded transcription via
Faster-Whisper (loaded only when speech transcription is explicitly requested).
"""

import os
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000
CHANNELS = 1
MODEL_NAME = "base"

# Global handle for lazy loading
_whisper_model = None

def get_whisper_model():
    """
    Lazy loader for Faster-Whisper model.
    Avoids loading the model into memory during dummy mode or text-only interviews.
    """
    global _whisper_model
    if _whisper_model is None:
        try:
            print("\n[AI Engine] Loading Faster-Whisper model ('base' on CPU)...")
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(
                MODEL_NAME,
                device="cpu",
                compute_type="int8"
            )
            print("[AI Engine] Faster-Whisper model loaded successfully.\n")
        except Exception as e:
            print(f"[Warning] Could not initialize Faster-Whisper: {e}")
            _whisper_model = None
    return _whisper_model

def record(audio_path: str) -> bool:
    """
    Interactively record audio from candidate microphone and save as a WAV file.
    Returns True if recording was successful, False if an error occurred.
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(audio_path)), exist_ok=True)

    print("\n--- Audio Recording ---")
    print("Press [Enter] to START recording your answer...")
    input()

    print(">>> Recording in progress... Speak into your microphone.")
    print("Press [Enter] to STOP recording...")

    recording = []

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}", flush=True)
        recording.append(indata.copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=callback
        ):
            input()
    except Exception as e:
        print(f"[Error] Microphone input device error: {e}")
        print("Please check your audio device or switch to text mode.")
        return False

    if not recording:
        print("[Warning] No audio recorded.")
        return False

    try:
        audio_array = np.concatenate(recording, axis=0)
        sf.write(audio_path, audio_array, SAMPLE_RATE)
        print(f"Recording saved successfully to '{audio_path}'.\n")
        return True
    except Exception as e:
        print(f"[Error] Failed to save audio file: {e}")
        return False

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe a recorded audio file to text using Faster-Whisper.
    Returns the transcribed text or empty string on error.
    """
    if not os.path.exists(audio_path):
        print(f"[Warning] Audio file '{audio_path}' does not exist.")
        return ""

    model = get_whisper_model()
    if model is None:
        print("[Error] Whisper model unavailable. Transcription skipped.")
        return ""

    try:
        print("Transcribing audio with Faster-Whisper...")
        segments, info = model.transcribe(audio_path, beam_size=5)
        transcript = "".join(segment.text for segment in segments).strip()
        print(f"Transcript: \"{transcript}\"")
        return transcript
    except Exception as e:
        print(f"[Error] Transcription failed: {e}")
        return ""

def speech_to_text(audio_path, candidates_data, corpus_data, candidate_id, question_id):
    """
    Transcribe audio and store results in both candidates_data and corpus_data
    without overwriting existing text or video fields.
    """
    transcript = transcribe_audio(audio_path)

    # 1. Update candidates_data safely
    if "candidates" not in candidates_data:
        candidates_data["candidates"] = {}
    if candidate_id not in candidates_data["candidates"]:
        candidates_data["candidates"][candidate_id] = {
            "metadata": {"candidate_id": candidate_id, "created_at": None},
            "responses": {}
        }
    cand_resp = candidates_data["candidates"][candidate_id].setdefault("responses", {})
    if question_id not in cand_resp:
        cand_resp[question_id] = {
            "text": {"answer": transcript},
            "speech": {"audio_path": audio_path, "transcript": transcript},
            "video": {"video_path": None}
        }
    else:
        cand_resp[question_id].setdefault("text", {})["answer"] = transcript
        cand_resp[question_id]["speech"] = {
            "audio_path": audio_path,
            "transcript": transcript
        }

    # 2. Update corpus_data safely
    if "corpus" not in corpus_data:
        corpus_data["corpus"] = {}
    if candidate_id not in corpus_data["corpus"]:
        corpus_data["corpus"][candidate_id] = {
            "metadata": {"candidate_id": candidate_id},
            "responses": {},
            "overall": {
                "text": {"features": {"statistics": {}, "sentiment": {}}},
                "speech": {"features": {}},
                "video": {"features": {}},
                "evaluation": {},
                "fusion": {}
            }
        }
    corp_resp = corpus_data["corpus"][candidate_id].setdefault("responses", {})
    if question_id not in corp_resp:
        corp_resp[question_id] = {
            "text": {"processed": {}, "features": {}},
            "speech": {"transcript": transcript, "features": {}},
            "video": {"features": {}},
            "evaluation": {},
            "fusion": {}
        }
    else:
        corp_resp[question_id].setdefault("speech", {})["transcript"] = transcript

    return candidates_data, corpus_data
