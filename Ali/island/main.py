"""
Dynamic island demo entrypoint.

Run from the `Ali` directory:
    python -m island.main

Right Shift: hold = Listening, release = Idle (if still listening).
Space (when the island window is focused): cycle Idle → Listening → Speaking → Idle.
"""

from __future__ import annotations

import sys

from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtWidgets import QApplication

from island.island_window import IslandWindow, configure_macos_window
from island.keyboard_bridge import KeyboardBridge
from island.voice_controller import VoiceController, VoiceState


class ShiftSignals(QObject):
    """Cross-thread safe: emit from pynput, slots run on the Qt main thread."""

    pressed = pyqtSignal()
    released = pyqtSignal()


def main() -> None:
    app = QApplication(sys.argv)

    controller = VoiceController()
    win = IslandWindow(controller)

    def on_state(state: VoiceState) -> None:
        if state == VoiceState.SPEAKING:
            QTimer.singleShot(2800, controller.mock_speak_done)
        # Re-anchor the NSWindow to the notch whenever the pill resizes.
        if sys.platform == "darwin":
            QTimer.singleShot(0, win.reposition_at_notch)

    controller.state_changed.connect(on_state)

    shift = ShiftSignals()
    shift.pressed.connect(controller.start_listen)
    shift.released.connect(controller.cancel_listen)

    bridge = KeyboardBridge(on_press=shift.pressed.emit, on_release=shift.released.emit)
    bridge.start()

    def cycle_demo() -> None:
        s = controller.state()
        if s == VoiceState.IDLE:
            controller.start_listen()
        elif s == VoiceState.LISTENING:
            controller.mock_transcript_ready()
        elif s == VoiceState.SPEAKING:
            controller.mock_speak_done()

    sc = QShortcut(QKeySequence(Qt.Key.Key_Space), win)
    sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
    sc.activated.connect(cycle_demo)

    win.show()
    if sys.platform == "darwin":
        configure_macos_window(win)

    code = app.exec()
    bridge.stop()
    sys.exit(code)


if __name__ == "__main__":
    main()
