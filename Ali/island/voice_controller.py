"""Voice UI state machine for the dynamic island (mock; no real audio)."""

from __future__ import annotations

from enum import Enum, auto

from PyQt6.QtCore import QObject, pyqtSignal


class VoiceState(Enum):
    IDLE = auto()
    LISTENING = auto()
    SPEAKING = auto()


class VoiceController(QObject):
    """Emits state_changed when VoiceState changes; UI stays dumb."""

    state_changed = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self._state = VoiceState.IDLE

    def state(self) -> VoiceState:
        return self._state

    def _set(self, new: VoiceState) -> None:
        if self._state != new:
            self._state = new
            self.state_changed.emit(new)

    def start_listen(self) -> None:
        self._set(VoiceState.LISTENING)

    def cancel_listen(self) -> None:
        if self._state == VoiceState.LISTENING:
            self._set(VoiceState.IDLE)

    def mock_transcript_ready(self) -> None:
        if self._state == VoiceState.LISTENING:
            self._set(VoiceState.SPEAKING)

    def mock_speak_done(self) -> None:
        if self._state == VoiceState.SPEAKING:
            self._set(VoiceState.IDLE)

    def demo_cycle_idle(self) -> None:
        """For Space shortcut: jump to Listening from Idle."""
        self._set(VoiceState.LISTENING)
