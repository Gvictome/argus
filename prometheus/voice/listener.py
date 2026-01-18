"""
Voice Listener - Microphone input with wake word detection
"""

import threading
from queue import Queue
from typing import Callable, Optional, List

# Lazy imports for optional dependencies
speech_recognition = None
whisper = None


def _ensure_speech_recognition():
    global speech_recognition
    if speech_recognition is None:
        try:
            import speech_recognition as sr
            speech_recognition = sr
        except ImportError:
            raise ImportError(
                "speech_recognition not installed. "
                "Install with: pip install SpeechRecognition pyaudio"
            )
    return speech_recognition


class VoiceListener:
    """
    Continuous microphone listener for voice commands

    Wake words: "Hey Prometheus", "Prometheus", "Senior Dev"

    Usage:
        listener = VoiceListener()
        listener.on_command(lambda cmd: print(f"Got: {cmd}"))
        listener.start()
    """

    WAKE_WORDS = ["prometheus", "senior dev", "hey prometheus"]

    def __init__(self, use_whisper: bool = False):
        """
        Initialize voice listener

        Args:
            use_whisper: Use local Whisper model instead of Google API
        """
        sr = _ensure_speech_recognition()

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.command_queue: Queue = Queue()
        self.is_listening = False
        self.is_awake = False
        self.use_whisper = use_whisper
        self._callbacks: List[Callable] = []
        self._listen_thread: Optional[threading.Thread] = None

        # Calibrate for ambient noise
        print("🎤 Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("🎤 Microphone ready")

    def start(self):
        """Start listening in background thread"""
        if self.is_listening:
            return

        self.is_listening = True
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        print("🎤 Prometheus listening... Say 'Hey Prometheus' to wake")

    def stop(self):
        """Stop listening"""
        self.is_listening = False
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)

    def _listen_loop(self):
        """Main listening loop"""
        sr = _ensure_speech_recognition()

        while self.is_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(
                        source,
                        timeout=5,
                        phrase_time_limit=10
                    )

                # Transcribe
                text = self._transcribe(audio)
                if not text:
                    continue

                text_lower = text.lower()

                # Check for wake word
                if not self.is_awake:
                    if any(wake in text_lower for wake in self.WAKE_WORDS):
                        self.is_awake = True
                        self._notify("🟢 Prometheus awake. Listening for commands...")

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
                        self._notify("🔴 Prometheus sleeping. Say wake word to activate.")
                    else:
                        self._process_command(text)

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                print(f"Listen error: {e}")

    def _transcribe(self, audio) -> Optional[str]:
        """Transcribe audio to text"""
        sr = _ensure_speech_recognition()

        try:
            if self.use_whisper:
                # Local Whisper model (offline)
                return self.recognizer.recognize_whisper(audio, model="base")
            else:
                # Google Speech Recognition (online, free)
                return self.recognizer.recognize_google(audio)

        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"API error: {e}")
            # Fallback to Whisper if available
            try:
                return self.recognizer.recognize_whisper(audio, model="base")
            except:
                return None

    def _process_command(self, command: str):
        """Queue command for processing"""
        print(f"🎤 Command: {command}")
        self.command_queue.put(command)

        for callback in self._callbacks:
            try:
                callback(command)
            except Exception as e:
                print(f"Callback error: {e}")

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
        self._notify("🟢 Prometheus awake (manual)")

    def sleep(self):
        """Manually put Prometheus to sleep"""
        self.is_awake = False
        self._notify("🔴 Prometheus sleeping (manual)")
