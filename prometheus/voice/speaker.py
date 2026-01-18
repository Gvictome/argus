"""
Text-to-Speech - Voice response output for Prometheus
"""

import platform
import subprocess
import threading
from typing import Optional


class TextToSpeech:
    """
    Cross-platform text-to-speech for Prometheus responses

    Supports:
    - Windows (SAPI)
    - macOS (say command)
    - Linux (espeak/festival)

    Usage:
        tts = TextToSpeech()
        tts.say("Hello, I am Prometheus")
    """

    def __init__(self, voice: Optional[str] = None, rate: int = 175):
        """
        Initialize TTS

        Args:
            voice: Voice name (platform-specific)
            rate: Speech rate (words per minute)
        """
        self.voice = voice
        self.rate = rate
        self.system = platform.system()
        self._speaking = False

    def say(self, text: str, block: bool = True):
        """
        Speak text aloud

        Args:
            text: Text to speak
            block: Wait for speech to complete
        """
        if block:
            self._speak(text)
        else:
            thread = threading.Thread(target=self._speak, args=(text,), daemon=True)
            thread.start()

    def _speak(self, text: str):
        """Internal speak method"""
        self._speaking = True

        try:
            if self.system == "Darwin":
                self._speak_macos(text)
            elif self.system == "Windows":
                self._speak_windows(text)
            elif self.system == "Linux":
                self._speak_linux(text)
            else:
                print(f"[TTS] {text}")

        except Exception as e:
            print(f"TTS error: {e}")
            print(f"[TTS] {text}")

        finally:
            self._speaking = False

    def _speak_macos(self, text: str):
        """Speak on macOS using 'say' command"""
        cmd = ["say"]
        if self.voice:
            cmd.extend(["-v", self.voice])
        cmd.append(text)
        subprocess.run(cmd, check=False)

    def _speak_windows(self, text: str):
        """Speak on Windows using SAPI"""
        # Escape quotes in text
        escaped_text = text.replace('"', '`"').replace("'", "`'")

        # PowerShell script for TTS
        rate = (self.rate - 175) // 25  # Convert to SAPI rate (-10 to 10)
        rate = max(-10, min(10, rate))

        ps_script = f'''
        Add-Type -AssemblyName System.Speech
        $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
        $synth.Rate = {rate}
        $synth.Speak("{escaped_text}")
        '''

        subprocess.run(
            ["powershell", "-Command", ps_script],
            check=False,
            capture_output=True
        )

    def _speak_linux(self, text: str):
        """Speak on Linux using espeak or festival"""
        # Try espeak first
        try:
            subprocess.run(
                ["espeak", "-s", str(self.rate), text],
                check=True,
                capture_output=True
            )
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Try festival
        try:
            subprocess.run(
                ["festival", "--tts"],
                input=text.encode(),
                check=True,
                capture_output=True
            )
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Try pico2wave
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_file = f.name

            subprocess.run(
                ["pico2wave", "-w", wav_file, text],
                check=True
            )
            subprocess.run(["aplay", wav_file], check=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        print(f"[TTS] {text}")

    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self._speaking

    def stop(self):
        """Stop current speech (best effort)"""
        if self.system == "Darwin":
            subprocess.run(["killall", "say"], capture_output=True)
        # Windows and Linux don't have easy stop mechanisms


# Singleton instance
tts = TextToSpeech()
