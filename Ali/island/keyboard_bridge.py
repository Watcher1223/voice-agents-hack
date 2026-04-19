"""pynput → Qt thread bridge for Right Shift (matches Ali/voice/capture.py)."""

from __future__ import annotations

from pynput import keyboard

TRIGGER_KEY = keyboard.Key.shift_r


class KeyboardBridge:
    """
    Press Right Shift → start_listen; release → cancel_listen while Listening.
    Pass callables that emit Qt signals so handlers run on the GUI thread.
    """

    def __init__(self, on_press, on_release) -> None:
        self._on_press = on_press
        self._on_release = on_release
        self._listener: keyboard.Listener | None = None

    def start(self) -> None:
        def on_press(key):
            if key == TRIGGER_KEY:
                self._on_press()

        def on_release(key):
            if key == TRIGGER_KEY:
                self._on_release()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
