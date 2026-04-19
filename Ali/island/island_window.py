"""Frameless floating island: top-center, drag, 50% opacity, optional macOS blur."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QGuiApplication
from PyQt6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from island.island_widget import IslandWidget, detect_notch_rect
from island.voice_controller import VoiceController


def _find_ns_window(win: QMainWindow):
    try:
        from AppKit import NSApplication
    except Exception:
        return None

    marker = "__ali_island__"
    prev = win.windowTitle()
    win.setWindowTitle(marker)
    QGuiApplication.processEvents()

    ns_app = NSApplication.sharedApplication()
    ns_win = None
    for w in ns_app.windows():
        try:
            if w.title() == marker:
                ns_win = w
                break
        except Exception:
            continue

    win.setWindowTitle(prev)
    return ns_win


def configure_macos_window(win: QMainWindow) -> None:
    """
    Position + level the NSWindow directly so the pill sits at the notch.
    Qt clamps frameless windows below the menu bar; Cocoa setFrame bypasses that.
    """
    try:
        from AppKit import NSColor, NSScreen

        ns_win = _find_ns_window(win)
        if ns_win is None:
            return

        ns_win.setOpaque_(False)
        ns_win.setBackgroundColor_(NSColor.clearColor())
        ns_win.setHasShadow_(False)
        # NSPopUpMenuWindowLevel = 101, above the menu bar (24).
        ns_win.setLevel_(101)
        # CanJoinAllSpaces | Stationary | FullScreenAuxiliary | IgnoresCycle
        ns_win.setCollectionBehavior_(1 | 16 | 256 | 64)
        ns_win.setIgnoresMouseEvents_(False)

        screen = NSScreen.mainScreen()
        if screen is None:
            return

        rect = detect_notch_rect()
        frame = screen.frame()  # Cocoa: origin bottom-left
        screen_h = float(frame.size.height)

        qt_w = float(win.frameGeometry().width())
        qt_h = float(win.frameGeometry().height())

        if rect is not None:
            notch_x, _y_from_top, notch_w, _notch_h = rect
            # Horizontally: center the Qt window on the notch.
            x = float(notch_x) + (float(notch_w) - qt_w) / 2.0
        else:
            x = (float(frame.size.width) - qt_w) / 2.0

        # Cocoa Y of the window's bottom-left = screen_h - qt_h (stick to top).
        y = screen_h - qt_h
        ns_win.setFrameOrigin_((x, y))
    except Exception as e:
        print(f"[island] macOS configure skipped: {e}")


class IslandWindow(QMainWindow):
    def __init__(self, controller: VoiceController) -> None:
        super().__init__()
        self._drag_pos = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setWindowOpacity(1.0)

        container = QWidget()
        container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        container.setAutoFillBackground(False)
        self.setCentralWidget(container)

        self._island = IslandWidget(controller)
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 12)
        lay.setSpacing(0)
        lay.addWidget(self._island)

        shadow = QGraphicsDropShadowEffect(self._island)
        shadow.setBlurRadius(16)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 110))
        self._island.setGraphicsEffect(shadow)

        self._place_top_center()

    def _place_top_center(self) -> None:
        """
        Pre-position via Qt so the first paint is roughly right; the real
        positioning (above the menu bar, aligned to the notch) happens in
        configure_macos_window after show().
        """
        self.adjustSize()
        screen = QGuiApplication.primaryScreen()
        if not screen:
            return
        full = screen.geometry()
        fw = self.frameGeometry().width()
        fh = self.frameGeometry().height()
        x = full.x() + (full.width() - fw) // 2
        y = full.y()
        self.setGeometry(x, y, fw, fh)

    def reposition_at_notch(self) -> None:
        configure_macos_window(self)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)
