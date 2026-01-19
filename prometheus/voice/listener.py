"""
Voice Listener - Microphone input with wake word detection

Uses sounddevice + whisper (no PyAudio required)
"""

import threading
import numpy as np
from queue import Queue
from typing import Callable, Optional, List
import tempfile
import wave
import os

# Lazy imports
sounddevice = None
whisper_model = None


def _ensure_sounddevice():
    global sounddevice
    if sounddevice is None:
        try:
            import sounddevice as sd
            sounddevice = sd
        except ImportError:
            raise ImportError(
                "sounddevice not installed. "
                "Install with: pip install sounddevice"
            )
    return sounddevice


def _ensure_whisper():
    global whisper_model
    if whisper_model is None:
        try:
            import whisper
            print("[voice] Loading Whisper model (first time may take a moment)...")
            whisper_model = whisper.load_model("base")
            print("[voice] Whisper model loaded")
        except ImportError:
            raise ImportError(
                "whisper not installed. "
                "Install with: pip install openai-whisper"
            )
    return whisper_model


class VoiceListener:
    """
    Continuous microphone listener for voice commands

    Uses sounddevice for recording and Whisper for transcription.
    No PyAudio required!

    Wake words: "Hey Prometheus", "Prometheus", "Senior Dev"

    Usage:
        listener = VoiceListener()
        listener.on_command(lambda cmd: print(f"Got: {cmd}"))
        listener.start()
    """

    WAKE_WORDS = ["prometheus", "senior dev", "hey prometheus"]

    # Audio settings
    SAMPLE_RATE = 16000  # Whisper expects 16kHz
    CHANNELS = 1
    DTYPE = np.float32

    def __init__(self, model_size: str = "base"):
        """
        Initialize voice listener

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.sd = _ensure_sounddevice()
        self.model_size = model_size
        self.model = None  # Lazy load

        self.command_queue: Queue = Queue()
        self.is_listening = False
        self.is_awake = False
        self._callbacks: List[Callable] = []
        self._listen_thread: Optional[threading.Thread] = None

        # Check microphone
        print("[voice] Checking microphone...")
        try:
            devices = self.sd.query_devices()
            default_input = self.sd.query_devices(kind='input')
            print(f"[voice] Using: {default_input['name']}")
        except Exception as e:
            print(f"[voice] Warning: {e}")

    def start(self):
        """Start listening in background thread"""
        if self.is_listening:
            return

        # Load whisper model now
        self.model = _ensure_whisper()

        self.is_listening = True
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        print("[voice] Prometheus listening... Say 'Hey Prometheus' to wake")

    def stop(self):
        """Stop listening"""
        self.is_listening = False
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)

    def _record_audio(self, duration: float = 5.0) -> np.ndarray:
        """Record audio from microphone"""
        frames = int(duration * self.SAMPLE_RATE)

        print("[voice] Listening...", end=" ", flush=True)
        audio = self.sd.rec(
            frames,
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=self.DTYPE
        )
        self.sd.wait()  # Wait for recording to finish
        print("done")

        return audio.flatten()

    def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio using Whisper"""
        if self.model is None:
            return None

        try:
            # Whisper expects float32 audio normalized to [-1, 1]
            audio = audio.astype(np.float32)

            # Transcribe
            result = self.model.transcribe(
                audio,
                language="en",
                fp16=False  # Use fp32 for CPU
            )

            text = result["text"].strip()
            if text:
                return text

        except Exception as e:
            print(f"[voice] Transcription error: {e}")

        return None

    def _listen_loop(self):
        """Main listening loop"""
        while self.is_listening:
            try:
                # Record audio chunk
                audio = self._record_audio(duration=5.0)

                # Skip if too quiet (silence detection)
                if np.abs(audio).max() < 0.01:
                    continue

                # Transcribe
                text = self._transcribe(audio)
                if not text:
                    continue

                print(f"[voice] Heard: {text}")
                text_lower = text.lower()

                # Check for wake word
                if not self.is_awake:
                    if any(wake in text_lower for wake in self.WAKE_WORDS):
                        self.is_awake = True
                        self._notify("[voice] Prometheus AWAKE. Listening for commands...")

                        # Extract command after wake word
                        for wake in self.WAKE_WORDS:
                            if wake in text_lower:
                                command = text_lower.split(wake, 1)[-1].strip()
                                if command:
                                    self._process_command(command)
                                break
                else:
                    # Already awake - process command directly
                    if "go to sleep" in text_lower or "stop listening" in text_lower:
                        self.is_awake = False
                        self._notify("[voice] Prometheus SLEEPING. Say wake word to activate.")
                    else:
                        self._process_command(text)

            except Exception as e:
                print(f"[voice] Listen error: {e}")

    def _process_command(self, command: str):
        """Queue command for processing"""
        print(f"[voice] Command: {command}")
        self.command_queue.put(command)

        for callback in self._callbacks:
            try:
                callback(command)
            except Exception as e:
                print(f"[voice] Callback error: {e}")

    def _notify(self, message: str):
        """Send notification"""
        print(message)

    def on_command(self, callback: Callable[[str], None]):
        """
        Register command callback

        Args:
            callback: Function called with command text
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """Remove a command callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_command(self, timeout: float = None) -> Optional[str]:
        """
        Get next command from queue

        Args:
            timeout: Max seconds to wait (None = forever)

        Returns:
            Command text or None if timeout
        """
        try:
            return self.command_queue.get(timeout=timeout)
        except:
            return None

    def wake(self):
        """Manually wake Prometheus"""
        self.is_awake = True
        self._notify("[voice] Prometheus awake (manual)")

    def sleep(self):
        """Manually put Prometheus to sleep"""
        self.is_awake = False
        self._notify("[voice] Prometheus sleeping (manual)")


# Quick test
if __name__ == "__main__":
    print("Testing voice listener...")
    listener = VoiceListener()
    listener.on_command(lambda cmd: print(f">>> COMMAND: {cmd}"))
    listener.start()

    print("\nSay 'Hey Prometheus' followed by a command.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        listener.stop()
        print("\nStopped.")
