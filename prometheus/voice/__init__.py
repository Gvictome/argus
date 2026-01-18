"""
Voice module for Prometheus

Components:
- VoiceListener: Microphone input with wake word detection
- TextToSpeech: Response output
"""

from prometheus.voice.listener import VoiceListener
from prometheus.voice.speaker import TextToSpeech

__all__ = ["VoiceListener", "TextToSpeech"]
