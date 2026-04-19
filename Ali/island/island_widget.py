"""Rounded pill content: collapsed vs expanded, mock waveform, state labels."""

from __future__ import annotations

import math

from PyQt6.QtCore import QEasingCurve, QRectF, Qt, QTimer, QVariantAnimation
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from island.voice_controller import VoiceController, VoiceState

W = 200
W_EXPANDED = 360
RADIUS = 10
EXPANDED_RADIUS = 22
COLLAPSED_H = 32
EXPANDED_H = 150
ANIM_MS = 280


def detect_notch_rect() -> tuple[int, int, int, int] | None:
    """Return (x, y_from_screen_top, width, height) of the notch in points, if present."""
    try:
        from AppKit import NSScreen

        screen = NSScreen.mainScreen()
        if screen is None:
            return None
        insets = screen.safeAreaInsets()
        top = float(insets.top)
        if top <= 0:
            return None
        frame = screen.frame()
        left = screen.auxiliaryTopLeftArea()
        right = screen.auxiliaryTopRightArea()
        notch_w = float(frame.size.width) - float(left.size.width) - float(right.size.width)
        if notch_w <= 0:
            return None
        x = int(round(float(left.size.width)))
        return x, 0, int(round(notch_w)), int(round(top))
    except Exception:
        return None


def detect_notch_size() -> tuple[int, int] | None:
    rect = detect_notch_rect()
    if rect is None:
        return None
    return rect[2], rect[3]

GREY_FILL = QColor(0, 0, 0, 255)
ACCENT = QColor(120, 220, 255, 235)


class IslandWidget(QWidget):
    def __init__(self, controller: VoiceController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._wave_phase = 0.0
        self._h_anim: QVariantAnimation | None = None
        self._w_anim: QVariantAnimation | None = None

        notch = detect_notch_size()
        self._collapsed_w = notch[0] if notch else W
        self._collapsed_h = notch[1] if notch else COLLAPSED_H
        self._collapsed_radius = min(RADIUS, self._collapsed_h)

        self.setFixedSize(self._collapsed_w, self._collapsed_h)

        self._title = QLabel("Ali")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet(
            "background: transparent; color: rgba(235,235,240,0.92); font-size: 13px; font-weight: 600;"
        )
        f = QFont()
        f.setPointSize(13)
        f.setWeight(QFont.Weight.DemiBold)
        self._title.setFont(f)

        self._subtitle = QLabel("Hold Right Shift to listen")
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle.setStyleSheet(
            "background: transparent; color: rgba(180,180,188,0.88); font-size: 11px;"
        )
        self._subtitle.hide()

        self._mock_line = QLabel("")
        self._mock_line.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._mock_line.setWordWrap(True)
        self._mock_line.setStyleSheet(
            "background: transparent; color: rgba(100,210,255,0.95); font-size: 12px;"
        )
        self._mock_line.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 4, 16, 8)
        layout.addWidget(self._title)
        layout.addWidget(self._subtitle)
        layout.addWidget(self._mock_line)
        layout.addStretch()

        self._wave_timer = QTimer(self)
        self._wave_timer.timeout.connect(self._tick_wave)
        self._wave_timer.setInterval(48)

        controller.state_changed.connect(self._on_state)
        self._on_state(controller.state())

    def _on_state(self, state: VoiceState) -> None:
        if state == VoiceState.IDLE:
            self._title.setText("")
            self._subtitle.hide()
            self._mock_line.hide()
            self._wave_timer.stop()
            self._animate_size(self._collapsed_w, self._collapsed_h)
        elif state == VoiceState.LISTENING:
            self._title.setText("Listening…")
            self._subtitle.setText("Release Right Shift to stop")
            self._subtitle.show()
            self._mock_line.hide()
            self._wave_timer.start()
            self._animate_size(W_EXPANDED, EXPANDED_H)
        elif state == VoiceState.SPEAKING:
            self._title.setText("Speaking")
            self._subtitle.hide()
            self._mock_line.setText('"What’s the weather in Tokyo?"')
            self._mock_line.show()
            self._wave_timer.start()
            self._animate_size(W_EXPANDED, EXPANDED_H)

    def _animate_size(self, target_w: int, target_h: int) -> None:
        self._h_anim = self._restart_anim(
            self._h_anim, self.height(), target_h, self._on_h_value
        )
        self._w_anim = self._restart_anim(
            self._w_anim, self.width(), target_w, self._on_w_value
        )

    def _restart_anim(self, old, start: int, end: int, on_value) -> QVariantAnimation:
        # stop() emits finished() → _on_size_anim_finished handles cleanup
        # (clears the ref and deleteLater). We deliberately do NOT call
        # disconnect() here: it races with the finished slot's deleteLater and
        # produces "wildcard call disconnects from destroyed signal" warnings.
        if old is not None:
            try:
                old.stop()
            except RuntimeError:
                pass

        anim = QVariantAnimation(self)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setDuration(ANIM_MS)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        anim.valueChanged.connect(on_value)
        anim.finished.connect(self._on_size_anim_finished)
        anim.start()
        return anim

    def _on_h_value(self, value: object) -> None:
        self._apply_size(self.width(), int(value))

    def _on_w_value(self, value: object) -> None:
        self._apply_size(int(value), self.height())

    def _on_size_anim_finished(self) -> None:
        sender = self.sender()
        if sender is self._h_anim:
            self._h_anim = None
        elif sender is self._w_anim:
            self._w_anim = None
        if sender is not None:
            sender.deleteLater()

    def _apply_size(self, w: int, h: int) -> None:
        w = max(self._collapsed_w, min(W_EXPANDED + 40, w))
        h = max(self._collapsed_h, min(EXPANDED_H + 40, h))
        self.setFixedSize(w, h)
        win = self.window()
        if win is not None:
            win.adjustSize()
            reposition = getattr(win, "reposition_at_notch", None)
            if callable(reposition):
                reposition()
            win.repaint()

    def _tick_wave(self) -> None:
        self._wave_phase += 0.22
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        # Blend radius as the pill expands: matches the notch when collapsed,
        # then softens to a chunkier rounded bottom as it grows.
        span = max(1, EXPANDED_H - self._collapsed_h)
        t = max(0.0, min(1.0, (h - self._collapsed_h) / span))
        r = self._collapsed_radius + (EXPANDED_RADIUS - self._collapsed_radius) * t

        path = self._notch_path(QRectF(0, 0, w, h), r)
        p.fillPath(path, GREY_FILL)

        if self._controller.state() in (VoiceState.LISTENING, VoiceState.SPEAKING):
            self._draw_wave(p)

    @staticmethod
    def _notch_path(rect: QRectF, r: float) -> QPainterPath:
        """Flat top, rounded bottom corners — matches the MacBook notch."""
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        r = min(r, w / 2.0, h)
        path = QPainterPath()
        path.moveTo(x, y)
        path.lineTo(x + w, y)
        path.lineTo(x + w, y + h - r)
        path.quadTo(x + w, y + h, x + w - r, y + h)
        path.lineTo(x + r, y + h)
        path.quadTo(x, y + h, x, y + h - r)
        path.closeSubpath()
        return path

    def _draw_wave(self, p: QPainter) -> None:
        cx = self.width() / 2
        base_y = self.height() - 28
        n = 10
        w = 4
        gap = 6
        total = n * w + (n - 1) * gap
        x0 = cx - total / 2
        for i in range(n):
            t = self._wave_phase + i * 0.45
            h = 4 + abs(math.sin(t)) * 18
            x = x0 + i * (w + gap)
            y = base_y - h
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(ACCENT)
            p.drawRoundedRect(QRectF(x, y, w, h), 2, 2)
