"""
PROMETHEUS - Senior Dev AI Orchestrator

Voice-activated master controller that manages
all projects, agents, and plugins.

Usage:
    python -m prometheus.main

Voice Commands:
    "Hey Prometheus" - Wake up
    "Create project heimdall" - Create new project
    "Switch to argus" - Change active project
    "Spawn error agent" - Start an agent
    "Install camera plugin" - Add a plugin
    "Go to sleep" - Deactivate
"""

import asyncio
from pathlib import Path

from prometheus.voice.listener import VoiceListener
from prometheus.voice.speaker import TextToSpeech
from prometheus.core.parser import CommandParser, CommandType
from prometheus.core.router import AgentRouter


BANNER = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ██████╗ ██████╗  ██████╗ ███╗   ███╗███████╗████████╗  ║
║   ██╔══██╗██╔══██╗██╔═══██╗████╗ ████║██╔════╝╚══██╔══╝  ║
║   ██████╔╝██████╔╝██║   ██║██╔████╔██║█████╗     ██║     ║
║   ██╔═══╝ ██╔══██╗██║   ██║██║╚██╔╝██║██╔══╝     ██║     ║
║   ██║     ██║  ██║╚██████╔╝██║ ╚═╝ ██║███████╗   ██║     ║
║   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝   ╚═╝     ║
║                                                           ║
║           🔥 Senior Dev AI Orchestrator v1.0 🔥           ║
║                                                           ║
║   Voice Commands:                                         ║
║   • "Hey Prometheus" - Wake up                            ║
║   • "Create project <name>" - New project                 ║
║   • "Spawn <type> agent" - Start agent                    ║
║   • "Go to sleep" - Deactivate                            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""


class Prometheus:
    """
    PROMETHEUS - Senior Dev AI Orchestrator

    The master controller that:
    - Listens for voice commands
    - Manages multiple projects
    - Controls specialized agents
    - Handles plugin systems
    """

    def __init__(self, workspace: Path = None, use_voice: bool = True):
        """
        Initialize Prometheus

        Args:
            workspace: Root directory for projects
            use_voice: Enable voice input (requires microphone)
        """
        self.workspace = workspace or Path.home() / "prometheus_workspace"
        self.use_voice = use_voice

        # Core components
        self.parser = CommandParser()
        self.router = AgentRouter(self.workspace)
        self.speaker = TextToSpeech()

        # Voice listener (optional)
        self.listener = None
        if use_voice:
            try:
                self.listener = VoiceListener()
            except ImportError as e:
                print(f"⚠️  Voice disabled: {e}")
                self.use_voice = False

        self.running = False

    async def start(self):
        """Start Prometheus"""
        print(BANNER)

        # Register command callback
        if self.listener:
            self.listener.on_command(self._on_command_sync)
            self.listener.start()
        else:
            print("⌨️  Text mode: Voice disabled")
            print("   Type commands below (or 'quit' to exit)\n")

        self.running = True

        # Main loop
        try:
            if self.use_voice:
                # Voice mode - just keep running
                while self.running:
                    await asyncio.sleep(0.1)
            else:
                # Text mode - read from stdin
                await self._text_input_loop()

        except KeyboardInterrupt:
            pass

        self.stop()

    async def _text_input_loop(self):
        """Handle text input when voice is disabled"""
        import sys

        while self.running:
            try:
                # Read input
                sys.stdout.write("prometheus> ")
                sys.stdout.flush()

                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                line = line.strip()

                if not line:
                    continue

                if line.lower() in ["quit", "exit", "q"]:
                    self.running = False
                    break

                await self._on_command(line)

            except EOFError:
                break

    def _on_command_sync(self, text: str):
        """Sync wrapper for async command handler"""
        asyncio.create_task(self._on_command(text))

    async def _on_command(self, text: str):
        """Handle a command (voice or text)"""
        print(f"\n🎤 Command: {text}")

        # Parse
        command = self.parser.parse(text)
        print(f"📝 Parsed: {command.type.value}")

        # Check for sleep/quit
        if command.type == CommandType.SLEEP:
            response = "Going to sleep. Goodbye."
            print(f"💬 {response}")
            self.speaker.say(response, block=False)

            if self.listener:
                self.listener.sleep()
            return

        # Execute
        result = await self.router.execute(command)

        # Build response
        if "error" in result:
            response = f"Error: {result['error']}"
        elif "message" in result:
            response = result["message"]
        else:
            response = "Done."

        print(f"💬 {response}")

        # Speak response
        self.speaker.say(response, block=False)

        # Print full result for debugging
        if result.get("success") or "error" not in result:
            for key, value in result.items():
                if key not in ["success", "message"]:
                    print(f"   {key}: {value}")

    def stop(self):
        """Stop Prometheus"""
        self.running = False

        if self.listener:
            self.listener.stop()

        print("\n👋 Prometheus shutting down...")


async def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Prometheus - Senior Dev AI Orchestrator")
    parser.add_argument(
        "--workspace", "-w",
        type=Path,
        default=None,
        help="Workspace directory for projects"
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable voice input (text mode only)"
    )

    args = parser.parse_args()

    prometheus = Prometheus(
        workspace=args.workspace,
        use_voice=not args.no_voice
    )

    await prometheus.start()


if __name__ == "__main__":
    asyncio.run(main())
