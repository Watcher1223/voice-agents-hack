"""
Liquid glass overlay — Apple-style frosted pill, top-center, expands downward.
"""

from __future__ import annotations

import datetime
import os
import queue
import subprocess
import sys
import threading
from typing import Callable

from PySide6.QtCore import (  # pyright: ignore[reportMissingImports]
    QEasingCurve, QPoint, QPropertyAnimation, QRect, QRectF, Qt, QObject,
    QTimer, Signal, Slot,
)
from PySide6.QtGui import (  # pyright: ignore[reportMissingImports]
    QBrush, QColor, QCursor, QFont, QGuiApplication, QImage, QLinearGradient,
    QPainter, QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import QApplication, QWidget  # pyright: ignore[reportMissingImports]
import math
import time

# #region agent log
def _dlog(loc: str, msg: str, data: dict, hid: str = "H2") -> None:
    try:
        import json as _j, os as _o, time as _t
        _p = "/Users/alspenceramitojr/Desktop/Ali/.cursor/debug-4ea166.log"
        _o.makedirs(_o.path.dirname(_p), exist_ok=True)
        with open(_p, "a") as _f:
            _f.write(_j.dumps({"sessionId":"4ea166","hypothesisId":hid,"location":loc,
                               "message":msg,"data":data,"timestamp":int(_t.time()*1000)})+"\n")
            _f.flush()
    except Exception:
        pass
# #endregion

# NOTE: `cv2` is deliberately NOT imported at module load time.
# `faster-whisper` pulls in PyAV, which bundles its own `libavdevice.dylib`
# registering `AVFFrameReceiver` / `AVFAudioReceiver`. `cv2` bundles the same
# classes. Having both loaded in the same process causes macOS to SIGTRAP
# ("zsh: trace trap") on the next AVFoundation call (even indirect ones from
# PyAudio/CoreAudio). Import cv2 lazily, only when the wake-scene actually
# opens the webcam.

W_WAKE    = 560
H_WAKE    = 200
CAM_W     = 160
CAM_H     = 160
USER_NAME = "Alspencer"

# ── Meeting mode geometry ─────────────────────────────────────────────────────
W_MEETING       = 560
H_MEETING_BASE  = 200    # height before any action items
H_ACTION_ROW    = 52     # height per action item row (fits 80×45 thumbnail)
MAX_ACTIONS_SHOWN = 5
MAX_TRANSCRIPT_CHARS = 320    # chars of rolling transcript shown
THUMB_W         = 80
THUMB_H         = 44


def _time_greeting() -> str:
    h = datetime.datetime.now().hour
    if h < 12:   return "Good morning"
    if h < 17:   return "Good afternoon"
    return "Good evening"


class _CamBridge(QObject):
    frame_ready = Signal(QImage)
    greeted     = Signal()

class _MacGlobalHotkeyBridge(QObject):
    """Marshals AppKit global key events to the Qt main thread."""

    backtick = Signal()
    right_option_down = Signal()
    right_option_up = Signal()


# Citation chip geometry (used by the disk-index answer panel).
CITATION_ROW_H = 26
CITATION_CHIP_PAD_X = 10
CITATION_CHIP_GAP = 8

# ── Design tokens (lifted from VoiceInk NotchRecorderView + MiniRecorderView,
#    and boring.notch sizing/matters.swift) ────────────────────────────────────
#
# Philosophy: pure black Dynamic-Island style body. Asymmetric corner radii
# (small top, larger bottom) make the overlay read as "hanging" from the menu
# bar. Inner elements stagger in with a ~90ms delay after the pill expands.

# Colors
FG          = QColor(255, 255, 255)        # pure white, full opacity
DIM         = QColor(255, 255, 255, 153)   # white @ 60% — VoiceInk subtext
FAINT       = QColor(255, 255, 255, 90)    # white @ 35% — VoiceInk inactive
DIVIDER_C   = QColor(255, 255, 255, 38)    # white @ 15% — VoiceInk Divider
RED         = QColor(255, 59, 48)          # systemRed — VoiceInk record color
YELLOW      = QColor(255, 214, 10)         # systemYellow
BLUE        = QColor(0, 122, 255)          # systemBlue
GREEN       = QColor(52, 199, 89)          # systemGreen
ERR         = QColor(255, 105, 97)         # soft red for errors
PROC_GRAY   = QColor(102, 102, 115)        # processing button fill

# Geometry — asymmetric radii (hanging-island look)
R_TOP_COMPACT  = 8    # VoiceInk compact top radius
R_BOT_COMPACT  = 16   # VoiceInk compact bottom radius
R_TOP_EXPANDED = 12   # VoiceInk live-text top
R_BOT_EXPANDED = 22   # VoiceInk live-text bottom
R_FLOATING     = 20   # MiniRecorderView compactCornerRadius (when floating)

W_PILL  = 340        # compact pill
W_FULL  = 560        # expanded history
H_PILL  = 44         # tightened: matches VoiceInk mainRow (notchH + 6)
MARGIN  = 0          # attach to menu bar edge (Dynamic Island anchor)
MARGIN_RIGHT = 16    # gap from right edge when docked-right
MAX_H   = 540
MAX_HIST = 8

# Spacing scale
PAD_H_COMPACT = 14
PAD_H_EXPANDED = 18
PAD_V = 10
STACK_SPACING = 10

# Audio visualizer (VoiceInk AudioVisualizer)
BAR_COUNT   = 13      # slightly fewer than VoiceInk's 15 for a tighter pill
BAR_W       = 3
BAR_SPACING = 2
BAR_H_MIN   = 3
BAR_H_MAX   = 22

# ── Docking ───────────────────────────────────────────────────────────────────
DOCK_TOP    = "top"
DOCK_RIGHT  = "right"

# ── Timing ────────────────────────────────────────────────────────────────────
PULSE_MS     = 40       # breath/bar tick — 25 fps
POLL_MS      = 40
AUTOHIDE_MS  = 5_000
SPRING_MS    = 420      # VoiceInk expand spring response × 1000
COLLAPSE_MS  = 450      # VoiceInk collapse spring response × 1000
STAGGER_MS   = 90       # VoiceInk inner-element delay


def _notch_path(w: int, h: int, r_top: int, r_bot: int) -> QPainterPath:
    """Asymmetric rounded rect: small top radii, larger bottom radii.

    This is the signature Dynamic Island shape — when anchored to the menu bar
    it reads as an extension hanging from the notch.
    """
    p = QPainterPath()
    p.moveTo(r_top, 0)
    p.lineTo(w - r_top, 0)
    p.arcTo(QRectF(w - 2 * r_top, 0, 2 * r_top, 2 * r_top), 90, -90)
    p.lineTo(w, h - r_bot)
    p.arcTo(QRectF(w - 2 * r_bot, h - 2 * r_bot, 2 * r_bot, 2 * r_bot), 0, -90)
    p.lineTo(r_bot, h)
    p.arcTo(QRectF(0, h - 2 * r_bot, 2 * r_bot, 2 * r_bot), 270, -90)
    p.lineTo(0, r_top)
    p.arcTo(QRectF(0, 0, 2 * r_top, 2 * r_top), 180, -90)
    p.closeSubpath()
    return p


def _bar_heights(t: float, amplitude: float, bar_count: int = BAR_COUNT) -> list[float]:
    """Return BAR_COUNT bar heights in [BAR_H_MIN, BAR_H_MAX].

    Default: VoiceInk-style wave — sinusoidal per-bar phase, boosted amplitude,
    center-weighted so the middle bars ride taller than the edges.

    ★ This is the most distinctive visual moment of the overlay. Tweak the wave
    shape (frequency, phases, center weight) to personalize the signature feel.
    """
    amp = max(0.0, min(1.0, amplitude)) ** 0.7
    center = (bar_count - 1) / 2
    heights = []
    for i in range(bar_count):
        phase = i * 0.4
        wave = 0.5 + 0.5 * math.sin(t * 8.0 + phase)
        center_boost = 1.0 - (abs(i - center) / center) * 0.4
        h = BAR_H_MIN + amp * wave * center_boost * (BAR_H_MAX - BAR_H_MIN)
        heights.append(h)
    return heights


USE_VIBRANCY = True    # Frosted-glass look via NSVisualEffectView.


def _apply_macos_overlay(win: QWidget) -> None:
    win._vibrancy_active = False  # type: ignore[attr-defined]
    try:
        from AppKit import NSApplication, NSColor, NSVisualEffectView  # type: ignore[reportMissingImports]

        marker = "__ali_overlay__"
        win.setWindowTitle(marker)
        QApplication.processEvents()

        ns_app = NSApplication.sharedApplication()
        ns_win = None
        for candidate in ns_app.windows():
            try:
                if candidate.title() == marker:
                    ns_win = candidate
                    break
            except Exception:
                continue

        if ns_win is not None:
            # NSPopUpMenuWindowLevel (101) sits above any normal app window
            # including Finder + Mail, so our overlay stays on top regardless
            # of which app is active.
            ns_win.setLevel_(101)
            # 1=CanJoinAllSpaces  8=Transient (no Mission Control card)
            # 64=IgnoresCycle (no Cmd+Tab entry)  256=FullScreenAuxiliary
            ns_win.setCollectionBehavior_(1 | 8 | 64 | 256)
            # Keep the overlay visible when our Python app deactivates
            # (Finder / Mail take focus). Without this, Qt.Tool windows
            # auto-hide on deactivation and the user has to click the Dock
            # icon to bring it back.
            try:
                ns_win.setHidesOnDeactivate_(False)
            except Exception:
                pass
            ns_win.setOpaque_(False)
            ns_win.setBackgroundColor_(NSColor.clearColor())

            win._ns_window = ns_win  # type: ignore[attr-defined]

            if not USE_VIBRANCY:
                # Dynamic-Island look: solid black body painted by Qt; skip the
                # NSVisualEffectView install entirely.
                win._vibrancy_active = False  # type: ignore[attr-defined]
                print("[overlay] dynamic-island mode (pure black, no vibrancy)")
            else:
                qt_view = ns_win.contentView()
                try:
                    frame_view = qt_view.superview()
                    if frame_view is None:
                        raise ValueError("no frame_view")
                    effect = NSVisualEffectView.alloc().initWithFrame_(qt_view.frame())
                    effect.setMaterial_(21)      # UnderWindowBackground — subtle
                    effect.setBlendingMode_(0)   # BehindWindow
                    effect.setState_(1)          # Active
                    effect.setAutoresizingMask_(18)
                    frame_view.addSubview_positioned_relativeTo_(effect, -1, qt_view)
                    win._ns_effect = effect  # type: ignore[attr-defined]
                    win._vibrancy_active = True  # type: ignore[attr-defined]
                    print("[overlay] liquid glass vibrancy active")
                except Exception as ve:
                    win._vibrancy_active = False  # type: ignore[attr-defined]
                    print(f"[overlay] vibrancy positioning skipped ({ve}) — using solid black")

        win.setWindowTitle("")
    except Exception as e:
        print(f"[overlay] vibrancy skipped: {e}")


def _open_citation_target(path: str) -> None:
    """Open a cited source when the user clicks its chip.

    * Filesystem paths open with macOS `open <path>` (uses the file's
      default app).
    * `ali://contacts/…` / `ali://calendar/…` / `ali://messages/…` open
      the matching macOS app. We can't easily deep-link to a specific
      contact or event through `open`, so we settle for opening the app
      itself — good enough as a jumping-off point.
    """
    if not path:
        return
    try:
        if path.startswith("ali://"):
            rest = path[len("ali://") :]
            source = rest.split("/", 1)[0]
            app_name = {
                "contacts": "Contacts",
                "calendar": "Calendar",
                "messages": "Messages",
            }.get(source)
            if app_name:
                subprocess.Popen(
                    ["open", "-a", app_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return
        subprocess.Popen(
            ["open", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _dlog(
            "overlay:open_citation",
            "opened cited file",
            {"path": path},
            "H2",
        )
    except Exception as exc:
        _dlog(
            "overlay:open_citation:error",
            "failed to open cited target",
            {"path": path, "err": str(exc)[:200]},
            "H2",
        )


def _paint_citation_chips(
    p: "QPainter",
    *,
    citations: list[dict],
    font: "QFont",
    pad_left: int,
    y: int,
    max_width: int,
) -> list[tuple["QRect", str]]:
    """Draw citation chips in a single row and return their hit rects.

    Each chip renders as an underlined blue label so it reads as a link.
    Returns a list of ``(chip_rect, path)`` pairs — mousePressEvent uses
    this to route clicks to the correct file.
    """
    p.save()
    p.setFont(font)
    fm = p.fontMetrics()
    hit_rects: list[tuple[QRect, str]] = []
    x = pad_left
    chip_h = 20
    chip_y = y + (CITATION_ROW_H - chip_h) // 2
    right_edge = pad_left + max_width
    for entry in citations:
        label = str(entry.get("label") or "").strip() or "(unnamed)"
        path = str(entry.get("path") or "")
        text_w = fm.horizontalAdvance(label)
        chip_w = text_w + CITATION_CHIP_PAD_X * 2
        if x + chip_w > right_edge and hit_rects:
            # No room for another chip on this row — stop (overflow).
            break
        chip_rect = QRect(x, chip_y, chip_w, chip_h)
        # Subtle pill background so the chip reads as tappable.
        bg = QColor(100, 210, 255, 34)
        border = QColor(100, 210, 255, 110)
        path_rect = QPainterPath()
        path_rect.addRoundedRect(chip_rect, chip_h / 2, chip_h / 2)
        p.fillPath(path_rect, bg)
        p.setPen(QPen(border, 1))
        p.drawPath(path_rect)
        # Label (underlined so it's recognisably a link even at a glance).
        link_font = QFont(font)
        link_font.setUnderline(True)
        p.setFont(link_font)
        p.setPen(BLUE if path else DIM)
        p.drawText(
            chip_rect,
            int(Qt.AlignmentFlag.AlignCenter),
            label,
        )
        p.setFont(font)
        if path:
            hit_rects.append((chip_rect, path))
        x += chip_w + CITATION_CHIP_GAP
    p.restore()
    return hit_rects


def _update_vibrancy_mask(win: QWidget) -> None:
    try:
        from Quartz import CGRectMake, CGPathCreateWithRoundedRect  # type: ignore[reportMissingImports]
        from Quartz.QuartzCore import CAShapeLayer  # type: ignore[reportMissingImports]
        effect = getattr(win, "_ns_effect", None)
        if effect is None:
            return
        w, h = win.width(), win.height()
        # Effect view lives in frame_view coords — keep its frame synced with
        # the Qt contentView's frame (they should be identical for frameless windows).
        try:
            ns_win = getattr(win, "_ns_window", None)
            if ns_win is not None:
                qt_frame = ns_win.contentView().frame()
                effect.setFrame_(qt_frame)
        except Exception:
            pass
        bounds = CGRectMake(0, 0, w, h)
        mask = CAShapeLayer.layer()
        # Use the larger bottom radius as a reasonable single-value mask fallback.
        # (Vibrancy is off by default in dynamic-island mode; this only runs if
        # USE_VIBRANCY = True.)
        path = CGPathCreateWithRoundedRect(bounds, R_BOT_EXPANDED, R_BOT_EXPANDED, None)
        mask.setPath_(path)
        effect.setWantsLayer_(True)
        effect.layer().setMask_(mask)
    except Exception:
        pass


class TranscriptionOverlay(QWidget):
    """Thread-safe: wake word calls schedule_wake_prompt() from a background thread."""

    _wake_listen_signal = Signal()

    def __init__(self, app: QApplication) -> None:
        super().__init__()
        self._app = app
        self._q: queue.Queue[tuple[str, str]] = queue.Queue()
        self._history: list[tuple[str, QColor, str]] = []
        # Clickable citation chips + their on-screen hit rectangles.
        # Populated when a `cited_paths` state is pushed; consulted by
        # mousePressEvent to open the underlying file when clicked.
        self._citations: list[dict] = []
        self._citation_hit_rects: list[tuple[QRect, str]] = []
        self._drag_offset: QPoint | None = None
        self._pulse_on = True
        self._recording = False
        self._prompt_armed = False
        self._pill_label = "Listening..."
        self._wake_capture_fn: Callable[[], None] | None = None
        # wake / call state
        self._wake_mode    = False
        self._wake_greeted = False
        self._wake_text    = ""
        self._cam_pixmap   = QPixmap()
        self._cam_bridge   = _CamBridge()
        self._cam_running  = False
        self._cam_bridge.frame_ready.connect(self._on_cam_frame)
        self._cam_bridge.greeted.connect(self._on_wake_greeted)

        # meeting capture state
        self._meeting_mode: bool = False
        self._meeting_transcript: str = ""   # rolling committed words (trimmed)
        self._meeting_interim: str = ""      # current partial phrase
        # (task, status, thumb_path). thumb_path = "" until screenshot_feed
        # captures one post-completion.
        self._meeting_actions: list[tuple[str, str, str]] = []
        # Pixmap cache keyed by thumb_path so we don't reload from disk
        # every paint.
        self._thumb_cache: dict[str, QPixmap] = {}

        self._dock_mode: str = DOCK_TOP

        # Click-to-confirm for ambient suggestions. When a confirmable
        # suggestion is active, left-click on the pill confirms and
        # right-click dismisses. Main.py registers the callbacks via
        # set_pending_confirm(); we clear on click or ambient_ack push.
        self._pending_confirm_cbs: tuple[Callable[[], None], Callable[[], None]] | None = None
        # Double-click the pill anywhere (outside close/citations) to
        # trigger push-to-talk — works without macOS Accessibility or
        # Input Monitoring permissions, unlike the backtick hotkey.
        self._on_double_click_ptt: Callable[[], None] | None = None

        self._font_label = QFont(".AppleSystemUIFont", 15, QFont.Weight.Bold)
        self._font_body  = QFont(".AppleSystemUIFont", 14)
        self._font_small = QFont(".AppleSystemUIFont", 12)
        self._font_close = QFont(".AppleSystemUIFont", 16, QFont.Weight.Medium)

        # Deliberately NOT using Qt.Tool — on macOS Qt auto-hides Tool
        # windows when the app deactivates, which cannot be overridden from
        # AppKit. Frameless + StaysOnTop + DoesNotAcceptFocus gives us a
        # regular borderless window that stays visible when focus shifts.
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)
        self.resize(W_PILL, H_PILL)
        self._reposition(W_PILL, H_PILL)
        # Show briefly so NSApp registers the NSWindow in its windows() list,
        # then immediately hide. _apply_macos_overlay searches that list.
        self.show()
        QApplication.processEvents()
        self.hide()
        _apply_macos_overlay(self)
        _update_vibrancy_mask(self)

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(POLL_MS)

        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse_tick)

        self._autohide_timer = QTimer(self)
        self._autohide_timer.setSingleShot(True)
        self._autohide_timer.timeout.connect(self.hide)

        self._wake_listen_signal.connect(self._on_wake_sequence)

        self._mac_hotkey_bridge = _MacGlobalHotkeyBridge(self)
        self._mac_hotkey_bridge.backtick.connect(self._on_global_backtick)
        self._mac_hotkey_bridge.right_option_down.connect(self._on_global_right_option_down)
        self._mac_hotkey_bridge.right_option_up.connect(self._on_global_right_option_up)
        self._macos_global_hotkey_monitor = None
        if (
            sys.platform == "darwin"
            and os.environ.get("ALI_ENABLE_HOTKEY") != "1"
            and os.environ.get("ALI_DISABLE_GLOBAL_HOTKEYS") != "1"
        ):
            self._install_macos_global_hotkeys()

    # ── Public ───────────────────────────────────────────────────────────────

    def push(self, state: str, text: str = "") -> None:
        self._q.put((state, text))

    def schedule_wake_prompt(self, start_capture: Callable[[], None]) -> None:
        """
        Conversational wake: show armed pill + pulse, play a listen chime on macOS,
        then invoke start_capture() (typically request_ptt_session_from_wake).
        Safe to call from non-Qt threads.
        """
        self._wake_capture_fn = start_capture
        self._wake_listen_signal.emit()

    @Slot()
    def _on_wake_sequence(self) -> None:
        if self._prompt_armed:
            return  # already armed — ignore duplicate wake trigger
        self._autohide_timer.stop()
        self._wake_mode = False
        self._prompt_armed = True
        self._recording = False
        self._pill_label = "Hi — I'm listening…"
        self._history.clear()
        self._dock_mode = DOCK_TOP
        self._pulse_on = True
        self._set_size(W_PILL, H_PILL)
        self._present()
        self._pulse_timer.start(PULSE_MS)
        self.update()
        QTimer.singleShot(0, self._play_wake_greeting)
        # Start capture almost immediately so wake feels instant.
        QTimer.singleShot(120, self._emit_wake_capture)

    def _play_wake_greeting(self) -> None:
        if sys.platform != "darwin":
            return
        try:
            from voice.speak import _voice
            proc = subprocess.Popen(
                ["/usr/bin/say", "-v", _voice(), "-r", "160", "Hi"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                from voice.speak import track_tts_process
                track_tts_process(proc)
            except Exception:
                pass
            # #region agent log
            _dlog("overlay:_play_wake_greeting", "wake greeting launched", {"pid": proc.pid}, "H2")
            # #endregion
        except Exception:
            # #region agent log
            _dlog("overlay:_play_wake_greeting:error", "wake greeting failed", {}, "H2")
            # #endregion

    def _emit_wake_capture(self) -> None:
        fn = self._wake_capture_fn
        self._wake_capture_fn = None
        if fn is not None:
            fn()

    def _install_macos_global_hotkeys(self) -> None:
        """
        Deliver `` ` `` and Right Option without pynput: on macOS, CGEventTap +
        Qt can SIGTRAP, so we use NSEvent.addGlobalMonitorForEventsMatchingMask.
        Requires Accessibility for the host app (Terminal, Cursor, etc.).
        """
        try:
            import AppKit
            from AppKit import NSEvent
        except ImportError:
            print("[overlay] AppKit unavailable — global hotkeys skipped", flush=True)
            return

        emitter = self._mac_hotkey_bridge
        # US/ANSI grave / backtick (kVK_ANSI_Grave); Right Option (kVK_RightOption)
        grave_code = 50
        right_option_code = 61

        def handler(event) -> None:  # type: ignore[no-untyped-def]
            try:
                et = event.type()
                kc = int(event.keyCode())
                if kc == grave_code and et == AppKit.NSEventTypeKeyDown:
                    if event.isARepeat():
                        return
                    emitter.backtick.emit()
                    return
                if kc == right_option_code:
                    if et == AppKit.NSEventTypeKeyDown:
                        emitter.right_option_down.emit()
                    elif et == AppKit.NSEventTypeKeyUp:
                        emitter.right_option_up.emit()
            except Exception:
                pass

        mask = (1 << AppKit.NSEventTypeKeyDown) | (1 << AppKit.NSEventTypeKeyUp)
        monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, handler)
        if monitor is None:
            print(
                "[overlay][warn] Global ` and Right Option inactive — grant "
                "Accessibility to this app in System Settings → Privacy & Security "
                "(or set ALI_ENABLE_HOTKEY=1 to use pynput; may crash with Qt).",
                flush=True,
            )
        else:
            self._macos_global_hotkey_monitor = monitor
            print(
                "[overlay] Global hotkeys: ` and Right Option (needs Accessibility)",
                flush=True,
            )

    @Slot()
    def _on_global_backtick(self) -> None:
        from voice.capture import invoke_backtick_callback

        invoke_backtick_callback()

    @Slot()
    def _on_global_right_option_down(self) -> None:
        from voice.capture import invoke_right_option_down

        invoke_right_option_down()

    @Slot()
    def _on_global_right_option_up(self) -> None:
        from voice.capture import invoke_right_option_up

        invoke_right_option_up()

    # ── Input ────────────────────────────────────────────────────────────────

    def set_on_double_click_ptt(self, fn: Callable[[], None]) -> None:
        """Register a handler that fires when the user double-clicks the
        pill (anywhere that isn't a close button or citation chip).
        Zero-permissions alternative to the backtick hotkey."""
        self._on_double_click_ptt = fn

    def mouseDoubleClickEvent(self, e) -> None:  # type: ignore[override]
        if e.button() != Qt.MouseButton.LeftButton:
            return
        x, y = e.position().x(), e.position().y()
        if self._hit_close(x, y):
            return
        # Skip if clicking a citation chip.
        for rect, _ in self._citation_hit_rects:
            if rect.contains(int(x), int(y)):
                return
        if self._on_double_click_ptt is not None:
            try:
                self._on_double_click_ptt()
            except Exception:
                pass

    def set_pending_confirm(
        self,
        on_confirm: Callable[[], None],
        on_dismiss: Callable[[], None],
    ) -> None:
        """Enter click-to-confirm mode: left-click → on_confirm, right-
        click → on_dismiss. Cleared when the click fires or when
        `clear_pending_confirm()` is called (e.g. on timeout)."""
        self._pending_confirm_cbs = (on_confirm, on_dismiss)

    def clear_pending_confirm(self) -> None:
        self._pending_confirm_cbs = None

    def mousePressEvent(self, e) -> None:  # type: ignore[override]
        # Click-to-confirm takes precedence over drag + close.
        if self._pending_confirm_cbs is not None:
            cbs = self._pending_confirm_cbs
            if e.button() == Qt.MouseButton.LeftButton and not self._hit_close(
                e.position().x(), e.position().y()
            ):
                self._pending_confirm_cbs = None
                try:
                    cbs[0]()
                except Exception:
                    pass
                return
            if e.button() == Qt.MouseButton.RightButton:
                self._pending_confirm_cbs = None
                try:
                    cbs[1]()
                except Exception:
                    pass
                return
        if e.button() == Qt.MouseButton.LeftButton:
            x = e.position().x()
            y = e.position().y()
            if self._hit_close(x, y):
                self._do_hide()
                return
            # Meeting-mode action row click → Quick-Look thumbnail + focus Chrome
            if self._meeting_mode:
                idx = self._hit_action_row(x, y)
                if idx is not None:
                    self._open_action_browser_view(idx)
                    return
            # Citation chip clicks open the underlying file / app.
            for rect, path in self._citation_hit_rects:
                if rect.contains(int(x), int(y)):
                    _open_citation_target(path)
                    return
            self._drag_offset = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent_cursor_hint(self, x: float, y: float) -> None:
        """Visual affordance: change cursor to pointing hand over citations."""
        for rect, _ in self._citation_hit_rects:
            if rect.contains(int(x), int(y)):
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                return
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def _hit_action_row(self, x: float, y: float) -> int | None:
        if not self._meeting_actions:
            return None
        # Mirrors _paint_meeting's layout:
        #   pad = 14 ; tx_y = 46 ; tx_h = 110 ; sep_y = 160 ; first ay = 186
        pad = 14
        first_ay = 46 + 110 + 4 + 26   # = 186
        if y < first_ay:
            return None
        # Clicks outside the row background rect (pad..w-pad) don't count.
        if x < pad or x > self.width() - pad:
            return None
        idx = int((y - first_ay) // H_ACTION_ROW)
        if 0 <= idx < len(self._meeting_actions):
            return idx
        return None

    def _open_action_browser_view(self, idx: int) -> None:
        """Open the captured thumbnail in Quick Look and focus Chrome."""
        if idx >= len(self._meeting_actions):
            return
        _task, _status, thumb_path = self._meeting_actions[idx]
        # Bring Chrome to front unconditionally — even without a thumb,
        # clicking a row is a natural "show me what's happening" gesture.
        try:
            from ui.screenshot_feed import focus_chrome
            focus_chrome()
        except Exception:
            pass
        if thumb_path:
            try:
                subprocess.Popen(
                    ["/usr/bin/qlmanage", "-p", thumb_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

    def mouseMoveEvent(self, e) -> None:  # type: ignore[override]
        if self._drag_offset and (e.buttons() & Qt.MouseButton.LeftButton):
            self.move(e.globalPosition().toPoint() - self._drag_offset)
        else:
            self.mouseMoveEvent_cursor_hint(e.position().x(), e.position().y())

    def mouseReleaseEvent(self, e) -> None:  # type: ignore[override]
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = None

    def resizeEvent(self, e) -> None:  # type: ignore[override]
        super().resizeEvent(e)
        _update_vibrancy_mask(self)

    def _hit_close(self, x: float, y: float) -> bool:
        cx, cy = self.width() - 23, 19
        return (x - cx) ** 2 + (y - cy) ** 2 <= 16 ** 2

    # ── Queue ────────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            while True:
                state, text = self._q.get_nowait()
                self._apply(state, text)
        except queue.Empty:
            pass

    # ── State ────────────────────────────────────────────────────────────────

    def _on_cam_frame(self, img: QImage) -> None:
        self._cam_pixmap = QPixmap.fromImage(img)
        self.update()

    def _on_wake_greeted(self) -> None:
        self._wake_greeted = True
        self.update()
        # Stay visible until user dismisses (× button or Space + Right Option)

    def _end_wake(self) -> None:
        self._cam_running = False
        self._wake_mode = False
        self._wake_greeted = False
        self._wake_text = ""
        self.hide()

    def _start_camera(self) -> None:
        import time
        self._cam_running = True

        def _prepare_tts(text: str) -> "str | None":
            import os, tempfile
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                try:
                    from openai import OpenAI  # type: ignore[reportMissingImports]
                    client = OpenAI(api_key=api_key)
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        path = f.name
                    with client.audio.speech.with_streaming_response.create(
                        model="tts-1-hd", voice="nova", input=text, speed=1.0,
                    ) as resp:
                        resp.stream_to_file(path)
                    return path
                except Exception as e:
                    print(f"[tts] OpenAI failed: {e} — falling back to say")
            return None

        def _play_tts(path: "str | None", text: str) -> None:
            import os, subprocess
            if path:
                subprocess.run(["afplay", path], check=True)
                os.unlink(path)
            else:
                subprocess.run(["say", "-v", "Flo (English (US))", "-r", "160", text])

        def _loop() -> None:
            import cv2  # type: ignore[reportMissingImports]
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            cap = cv2.VideoCapture(0)
            face_first: float | None = None
            greeted = False
            tts_started = False
            greeting = (
                f"{_time_greeting()}, {USER_NAME}. "
                "While you were asleep I've been busy — "
                "I found some great opportunities and took care of a few things. "
                "Let me walk you through them."
            )

            while self._cam_running:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(grey, 1.1, 5, minSize=(60, 60))

                if len(faces) > 0 and not greeted and not tts_started:
                    if face_first is None:
                        face_first = time.time()
                    if time.time() - face_first >= 1.2:
                        greeted = True
                        tts_started = True
                        def _greet_sync(g=greeting) -> None:
                            # Pre-generate audio first, then show text + play simultaneously
                            audio_path = _prepare_tts(g)
                            self._cam_bridge.greeted.emit()
                            threading.Thread(target=_play_tts, args=(audio_path, g), daemon=True).start()
                        threading.Thread(target=_greet_sync, daemon=True).start()
                elif not greeted and not tts_started:
                    face_first = None

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fh, fw = rgb.shape[:2]
                img = QImage(rgb.data, fw, fh, fw * 3, QImage.Format.Format_RGB888).copy()
                self._cam_bridge.frame_ready.emit(img)

            cap.release()

        threading.Thread(target=_loop, daemon=True).start()

    def _apply(self, state: str, text: str) -> None:
        self._autohide_timer.stop()
        self._pulse_timer.stop()

        if state == "hidden":
            self._dock_mode = DOCK_TOP
            self._do_hide()
            return

        if state == "wake":
            if self._wake_mode:
                return  # already in wake mode — ignore duplicate trigger
            self._cam_running = False  # stop any lingering camera thread
            self._dock_mode = DOCK_TOP
            self._wake_mode = True
            self._wake_greeted = False
            self._wake_text = ""
            self._reposition(W_WAKE, H_WAKE)
            self.show()
            self.raise_()
            self._reassert_window_level()
            self._start_camera()
            return

        # ── Meeting mode states ──────────────────────────────────────────────
        if state == "meeting_start":
            self._meeting_mode       = True
            self._meeting_transcript = ""
            self._meeting_interim    = ""
            self._meeting_actions    = []
            self._dock_mode          = DOCK_RIGHT
            self._recording          = False
            self._prompt_armed       = False
            h = H_MEETING_BASE
            self._reposition(W_MEETING, h)
            self._present()
            self._pulse_timer.start(PULSE_MS)
            self.update()
            return

        if state == "meeting_interim":
            # Frequent — just update interim text and repaint; don't resize
            self._meeting_interim = text
            self.update()
            return

        if state == "meeting_final":
            # Committed utterance: append to rolling transcript, clear interim
            self._meeting_interim = ""
            combined = (self._meeting_transcript + " " + text).strip()
            # Keep last MAX_TRANSCRIPT_CHARS visible
            if len(combined) > MAX_TRANSCRIPT_CHARS:
                combined = "…" + combined[-MAX_TRANSCRIPT_CHARS:]
            self._meeting_transcript = combined
            self.update()
            return

        if state == "meeting_action":
            self._meeting_actions.append((text, "Queued", ""))
            self._meeting_actions = self._meeting_actions[-MAX_ACTIONS_SHOWN:]
            n = len(self._meeting_actions)
            h = H_MEETING_BASE + n * H_ACTION_ROW + 12
            self._reposition(W_MEETING, h)
            self.update()
            return

        if state == "meeting_action_update":
            # text = "task_description|status"
            # status may be the literal "thumb:<path>" — in that case we
            # attach the screenshot to the row without changing its status.
            if "|" in text:
                task, status = text.split("|", 1)
                is_thumb = status.startswith("thumb:")
                for i, (t, st, th) in enumerate(self._meeting_actions):
                    if t == task:
                        if is_thumb:
                            self._meeting_actions[i] = (t, st, status[len("thumb:"):])
                        else:
                            self._meeting_actions[i] = (t, status, th)
                        break
            self.update()
            return

        if state == "meeting_stop":
            self._meeting_mode       = False
            self._meeting_transcript = ""
            self._meeting_interim    = ""
            self._meeting_actions    = []
            self._pulse_timer.stop()
            self._do_hide()
            return

        if state == "recording":
            self._dock_mode = DOCK_TOP
            self._history.clear()
            self._prompt_armed = False
            self._recording = True
            self._pill_label = "Listening..."
            self._pulse_on = True
            if not self._wake_mode:
                self._set_size(W_PILL, H_PILL)
            self.show()
            self.raise_()
            self._reassert_window_level()
            self._pulse_timer.start(PULSE_MS)
            self.update()
            return

        self._recording = False
        self._prompt_armed = False

        if state == "transcribing":
            pass  # no-op — don't show "Transcribing…" in history
        elif state == "transcript":
            # New command: reset history and dock back to top-center.
            # Also clear any lingering citations from the previous turn.
            self._history.clear()
            self._citations = []
            self._citation_hit_rects = []
            self._dock_mode = DOCK_TOP
            self._history.append((text, FG, "user"))
        elif state == "intent":
            pass  # skip intent label — action line already conveys this
        elif state == "action":
            self._history.append((text, FG, "assistant"))
            self._dock_mode = DOCK_RIGHT
        elif state == "revealed":
            label = f"Revealed: {text}" if text else "Revealed in Finder"
            self._history.append((label, GREEN, "assistant"))
            self._dock_mode = DOCK_RIGHT
        elif state == "done":
            self._history.append(("✓  Done", GREEN, "assistant"))
            # No autohide — stay visible beside the app until next command
        elif state == "error":
            self._history.append((text or "Error", ERR, "assistant"))
        elif state == "assistant":
            self._history.append((text, FG, "assistant"))
            # No autohide — stay visible until next command or × dismiss
        elif state == "ambient_confirm":
            # Destructive ambient suggestion awaiting user's 'yes'/backtick.
            # Yellow signals "needs input"; text should already read like a
            # prompt ("Send email to Hanzi?  say 'yes' to confirm").
            self._history.append((text or "Proceed?", YELLOW, "assistant"))
        elif state == "ambient_ack":
            # Follow-up after a confirmation — either ✓ executed or ✗ skipped.
            color = GREEN if text.startswith("✓") else DIM
            self._history.append((text, color, "assistant"))
        elif state == "cited_paths":
            # text is a JSON-encoded list of {label, path} dicts. Store them
            # so paintEvent can render clickable chips.
            import json as _json
            try:
                items = _json.loads(text or "[]")
            except _json.JSONDecodeError:
                items = []
            self._citations = [
                {"label": str(i.get("label", "")), "path": str(i.get("path", ""))}
                for i in items
                if isinstance(i, dict) and i.get("path")
            ]
            self._citation_hit_rects = []
        elif state == "cited":
            # Legacy text-only citation — treat as a single chip without
            # a path (not clickable, kept for backward compatibility).
            self._citations = [{"label": text, "path": ""}]
            self._citation_hit_rects = []
        else:
            self._history.append((text, FG, "assistant"))

        self._history = self._history[-MAX_HIST:]
        self._set_size(W_FULL, self._calc_height())
        self._present()
        self.update()

    def _calc_height(self) -> int:
        PAD_TOP = 18
        PAD_BOT = 18
        SEP = 10      # separator under user transcript
        LINE_H = 26   # height per wrapped line of body text
        SMALL_H = 22  # height for user transcript line
        h = PAD_TOP
        for text, _, kind in self._history:
            if kind == "user":
                h += SMALL_H + SEP
            else:
                lines = max(1, (len(text) + 46) // 47)
                h += lines * LINE_H + 6
        if self._citations:
            # Citation chips render in one horizontal row under the body.
            h += CITATION_ROW_H + 6
        h += PAD_BOT
        return min(MAX_H, max(H_PILL, h))

    def _set_size(self, w: int, h: int, *, animated: bool = True) -> None:
        self._reposition(w, h, animated=animated)

    def _reposition(self, w: int, h: int, *, animated: bool = True) -> None:
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        if not screen:
            self.resize(w, h)
            return

        geo = screen.availableGeometry()
        if self._dock_mode == DOCK_RIGHT:
            x = geo.right() - w - MARGIN_RIGHT
            y = geo.top() + 8   # small float gap when docked-right
        else:
            # DOCK_TOP: anchor to the menu bar so the pill reads as a Dynamic
            # Island extension hanging from the notch.
            x = geo.center().x() - w // 2
            y = geo.top() + MARGIN
        target = QRect(x, y, w, h)

        if not animated or not self.isVisible():
            self.setGeometry(target)
            return

        # Spring-ish expand / collapse via OutBack easing. VoiceInk uses a real
        # SwiftUI spring; QPropertyAnimation doesn't have one, but OutBack with
        # a small overshoot reads similarly.
        anim = getattr(self, "_geo_anim", None)
        if anim is None:
            anim = QPropertyAnimation(self, b"geometry")
            self._geo_anim = anim  # type: ignore[attr-defined]
        else:
            anim.stop()

        collapsing = (w * h) < (self.width() * self.height())
        if collapsing:
            anim.setDuration(COLLAPSE_MS)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        else:
            anim.setDuration(SPRING_MS)
            curve = QEasingCurve(QEasingCurve.Type.OutBack)
            curve.setOvershoot(1.2)  # subtle — ≈ spring damping 0.80
            anim.setEasingCurve(curve)
        anim.setStartValue(self.geometry())
        anim.setEndValue(target)
        anim.start()

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)

        w, h = self.width(), self.height()

        # Choose radii based on mode. Dynamic-island compact = small top / big
        # bottom; expanded history panel = larger top / larger bottom.
        if self._meeting_mode or self._wake_mode or not (self._recording or self._prompt_armed):
            r_top, r_bot = R_TOP_EXPANDED, R_BOT_EXPANDED
        else:
            r_top, r_bot = R_TOP_COMPACT, R_BOT_COMPACT

        # When docked right (floating), symmetric radii look better — the pill
        # is no longer "hanging" from the menu bar.
        if self._dock_mode == DOCK_RIGHT:
            r_top = r_bot = R_FLOATING

        shell = _notch_path(w, h, r_top, r_bot)

        if self._wake_mode:
            self._paint_wake(p, w, h, shell)
            return

        if self._meeting_mode:
            self._paint_black_body(p, shell, w, h)
            self._paint_meeting(p, w, h)
            return

        self._paint_black_body(p, shell, w, h)
        if self._recording or self._prompt_armed:
            self._paint_pill(p)
        else:
            self._paint_expanded(p)

    def _paint_pill(self, p: QPainter) -> None:
        """Dynamic-Island-style compact pill: breathing red dot on the left,
        live audio-bar visualizer on the right. Mirrors VoiceInk's notch mode."""
        w, h = self.width(), self.height()
        cy = h // 2

        # ── Left: breathing record dot ───────────────────────────────────────
        # Continuous sin-wave opacity instead of binary blink. More iOS-like.
        t = time.monotonic()
        breath = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(t * 3.2))  # 0.55..1.0
        dot_color = QColor(RED)
        dot_color.setAlphaF(breath)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(dot_color)
        p.drawEllipse(QPoint(PAD_H_COMPACT + 6, cy), 5, 5)

        # ── Center: label ────────────────────────────────────────────────────
        p.setPen(FG)
        p.setFont(self._font_label)
        label_x = PAD_H_COMPACT + 20
        label_w = w - PAD_H_COMPACT - 20 - (BAR_COUNT * (BAR_W + BAR_SPACING) + PAD_H_COMPACT)
        p.drawText(
            QRect(label_x, cy - 10, label_w, 20),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self._pill_label,
        )

        # ── Right: live audio bars ───────────────────────────────────────────
        bar_group_w = BAR_COUNT * BAR_W + (BAR_COUNT - 1) * BAR_SPACING
        bar_x0 = w - PAD_H_COMPACT - bar_group_w
        # Synthetic amplitude until real mic meter is wired: breathe with the dot.
        amp = 0.4 + 0.6 * breath
        heights = _bar_heights(t, amp)
        p.setBrush(QColor(255, 255, 255, 217))  # white @ 85% — VoiceInk bar color
        p.setPen(Qt.PenStyle.NoPen)
        for i, bh in enumerate(heights):
            bh_clamped = max(BAR_H_MIN, min(BAR_H_MAX, bh))
            bx = bar_x0 + i * (BAR_W + BAR_SPACING)
            by = cy - bh_clamped / 2
            p.drawRoundedRect(QRectF(bx, by, BAR_W, bh_clamped), BAR_W / 2, BAR_W / 2)

    def _paint_black_body(
        self, p: QPainter, shell: QPainterPath, w: int, h: int
    ) -> None:
        """Body fill. Two modes:
          • vibrancy active → translucent dark tint so NSVisualEffectView blur
            shows through, plus a soft white edge. (Original "liquid glass".)
          • vibrancy off    → pure black Dynamic-Island look, no border.
        """
        vibrancy = getattr(self, "_vibrancy_active", False)
        if not vibrancy:
            p.fillPath(shell, QColor(0, 0, 0))
            return

        # Soft drop shadow for bloom.
        for offset, alpha in ((4, 14), (2, 8)):
            s = QPainterPath()
            s.addRoundedRect(QRect(offset // 2, offset, w - offset, h), R_FLOATING, R_FLOATING)
            p.fillPath(s, QColor(0, 0, 0, alpha))

        # Translucent dark tint on top of the vibrancy blur.
        p.fillPath(shell, QColor(18, 18, 22, 145))

        # Glassy top-to-bottom edge.
        border = QLinearGradient(0, 0, 0, h)
        border.setColorAt(0.0, QColor(255, 255, 255, 68))
        border.setColorAt(0.5, QColor(210, 220, 240, 44))
        border.setColorAt(1.0, QColor(156, 168, 188, 30))
        p.setPen(QPen(QBrush(border), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(shell)

    def _paint_meeting(self, p: QPainter, w: int, h: int) -> None:
        """Meeting capture overlay: live transcript + action items queue."""
        pad = PAD_H_EXPANDED

        # ── Header ────────────────────────────────────────────────────────────
        t = time.monotonic()
        breath = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(t * 3.2))
        dot_col = QColor(RED); dot_col.setAlphaF(breath)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(dot_col)
        p.drawEllipse(QPoint(pad + 4, 22), 5, 5)

        p.setPen(FG)
        p.setFont(self._font_label)
        p.drawText(QRect(pad + 18, 10, w - 90, 24), Qt.AlignmentFlag.AlignVCenter,
                   "Meeting Capture")

        p.setPen(DIM)
        p.setFont(self._font_small)
        p.drawText(QRect(w - 72, 10, 60, 24),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   "say stop")

        # Divider
        p.setPen(QPen(DIVIDER_C, 1))
        p.drawLine(QPoint(pad, 38), QPoint(w - pad, 38))

        # ── Transcript area ───────────────────────────────────────────────────
        tx_y   = 46
        tx_h   = 110
        tx_w   = w - pad * 2

        # Committed transcript (dim)
        p.setPen(DIM)
        p.setFont(self._font_small)
        _flags = int(Qt.TextFlag.TextWordWrap) | int(Qt.AlignmentFlag.AlignLeft) | int(Qt.AlignmentFlag.AlignBottom)
        p.drawText(QRect(pad, tx_y, tx_w, tx_h - 22), _flags,
                   self._meeting_transcript or "Listening to meeting…")

        # Interim text (live, brighter) — breathe the alpha on a ~1.6s cycle
        if self._meeting_interim:
            interim_alpha = int(170 + 70 * (0.5 + 0.5 * math.sin(t * 3.9)))
            p.setPen(QColor(246, 246, 250, interim_alpha))
            p.setFont(self._font_body)
            p.drawText(QRect(pad, tx_y + tx_h - 26, tx_w, 22),
                       int(Qt.AlignmentFlag.AlignLeft) | int(Qt.AlignmentFlag.AlignVCenter),
                       self._meeting_interim)
        else:
            # Cursor blink at ~1 Hz — decoupled from the 25 fps repaint tick
            if int(t * 2) % 2 == 0:
                p.setPen(QColor(100, 210, 255, 180))
                p.setFont(self._font_body)
                p.drawText(QRect(pad, tx_y + tx_h - 26, 20, 22),
                           Qt.AlignmentFlag.AlignLeft, "▌")

        if not self._meeting_actions:
            return

        # Divider before actions
        sep_y = tx_y + tx_h + 4
        p.setPen(QPen(DIVIDER_C, 1))
        p.drawLine(QPoint(pad, sep_y), QPoint(w - pad, sep_y))

        p.setPen(DIM)
        p.setFont(self._font_small)
        p.drawText(QRect(pad, sep_y + 4, 120, 18), Qt.AlignmentFlag.AlignLeft, "ACTION ITEMS")

        # ── Action items ──────────────────────────────────────────────────────
        ay = sep_y + 26
        for task, status, thumb_path in self._meeting_actions:
            # "done:$189 on Delta" → green, "✓ $189 on Delta"
            # "done"              → green, "✓ Done"
            # "Running"           → blue (pulses via _pulse_on)
            # "error"             → red
            sl = status.lower()
            if sl.startswith("done:"):
                sc    = GREEN
                label = "✓ " + status[5:][:32]
            elif sl == "done":
                sc    = GREEN
                label = "✓ Done"
            elif sl == "running":
                sc    = BLUE if self._pulse_on else QColor(100, 180, 255, 120)
                label = "Running…"
            elif sl == "error":
                sc    = ERR
                label = "✗ Error"
            elif sl == "queued":
                sc    = DIM
                label = "Queued"
            else:
                sc    = DIM
                label = status[:20]

            # Subtle row background
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 14))
            p.drawRoundedRect(pad, ay, w - pad * 2, H_ACTION_ROW - 4, 10, 10)

            # Thumbnail (if captured) — painted on the right before the status label.
            thumb_right = w - pad - 8
            status_right_edge = thumb_right
            if thumb_path:
                pix = self._thumb_cache.get(thumb_path)
                if pix is None or pix.isNull():
                    loaded = QPixmap(thumb_path)
                    if not loaded.isNull():
                        pix = loaded
                        self._thumb_cache[thumb_path] = loaded
                if pix is not None and not pix.isNull():
                    thumb_x = thumb_right - THUMB_W
                    thumb_y = ay + (H_ACTION_ROW - 4 - THUMB_H) // 2
                    thumb_path_rect = QPainterPath()
                    thumb_path_rect.addRoundedRect(
                        QRect(thumb_x, thumb_y, THUMB_W, THUMB_H), 6, 6
                    )
                    p.save()
                    p.setClipPath(thumb_path_rect)
                    scaled = pix.scaled(
                        THUMB_W, THUMB_H,
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    ox = (scaled.width()  - THUMB_W) // 2
                    oy = (scaled.height() - THUMB_H) // 2
                    p.drawPixmap(thumb_x - ox, thumb_y - oy, scaled)
                    p.restore()
                    # Thin border so the thumb reads as a clickable tile.
                    p.setPen(QPen(QColor(255, 255, 255, 80), 1))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawPath(thumb_path_rect)
                    status_right_edge = thumb_x - 6

            p.setPen(FG)
            p.setFont(self._font_small)
            # Task label — leave room for status and thumb on the right.
            task_right_pad = (w - status_right_edge) + 110
            p.drawText(QRect(pad + 10, ay + 2, w - task_right_pad, H_ACTION_ROW - 6),
                       int(Qt.AlignmentFlag.AlignLeft) | int(Qt.AlignmentFlag.AlignVCenter),
                       task)

            p.setPen(sc)
            p.drawText(QRect(status_right_edge - 100, ay + 2, 100, H_ACTION_ROW - 6),
                       int(Qt.AlignmentFlag.AlignRight) | int(Qt.AlignmentFlag.AlignVCenter),
                       label)

            ay += H_ACTION_ROW

    def _paint_wake(self, p: QPainter, w: int, h: int, shell: QPainterPath) -> None:
        # Low alpha — let desktop vibrancy bleed through the whole panel
        p.fillPath(shell, QColor(18, 18, 22, 60))
        border = QLinearGradient(0, 0, 0, h)
        border.setColorAt(0, QColor(255, 255, 255, 70))
        border.setColorAt(1, QColor(255, 255, 255, 20))
        p.setPen(QPen(QBrush(border), 1.2))
        p.drawPath(shell)

        pad = 16
        cam_x, cam_y = pad, (h - CAM_H) // 2

        # Camera feed (rounded)
        if not self._cam_pixmap.isNull():
            cam_path = QPainterPath()
            cam_path.addRoundedRect(QRect(cam_x, cam_y, CAM_W, CAM_H), 14, 14)
            p.setClipPath(cam_path)
            scaled = self._cam_pixmap.scaled(
                CAM_W, CAM_H,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            ox = (scaled.width()  - CAM_W) // 2
            oy = (scaled.height() - CAM_H) // 2
            p.drawPixmap(cam_x - ox, cam_y - oy, scaled)
            p.setClipping(False)
            p.setPen(QPen(QColor(255, 255, 255, 50), 1))
            p.drawPath(cam_path)
        else:
            p.fillRect(QRect(cam_x, cam_y, CAM_W, CAM_H), QColor(40, 40, 44))

        # Recording indicator (top-right corner when listening)
        if self._recording:
            breath = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(time.monotonic() * 3.2))
            dot_color = QColor(RED); dot_color.setAlphaF(breath)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(dot_color)
            p.drawEllipse(QPoint(w - 20, 16), 5, 5)
            p.setPen(DIM)
            p.setFont(self._font_small)
            p.drawText(QRect(w - 110, 8, 82, 16), Qt.AlignmentFlag.AlignRight, "Listening…")

        # Text area (right of camera)
        tx = cam_x + CAM_W + pad
        tw = w - tx - pad

        if not self._wake_greeted:
            p.setPen(DIM)
            p.setFont(self._font_small)
            p.drawText(QRect(tx, cam_y + 10, tw, 30), Qt.AlignmentFlag.AlignLeft, "Ali is watching...")
            # progress dots
            p.setPen(FG)
            p.setFont(self._font_body)
            p.drawText(QRect(tx, cam_y + 50, tw, 30), Qt.AlignmentFlag.AlignLeft, "Detecting face...")
        else:
            p.setPen(GREEN)
            p.setFont(self._font_label)
            p.drawText(QRect(tx, cam_y + 8, tw, 28), Qt.AlignmentFlag.AlignLeft,
                       f"{_time_greeting()}, {USER_NAME}!")
            p.setPen(FG)
            p.setFont(self._font_small)
            p.drawText(
                QRect(tx, cam_y + 44, tw, CAM_H - 50),
                int(Qt.TextFlag.TextWordWrap),
                "While you were asleep I've been busy — I found great opportunities and took care of things.",
            )

    def _paint_expanded(self, p: QPainter) -> None:
        w, h = self.width(), self.height()
        pad = PAD_H_EXPANDED

        # Subtle × button
        p.setPen(FAINT)
        p.setFont(self._font_close)
        p.drawText(QRect(w - 30, 8, 20, 20), Qt.AlignmentFlag.AlignCenter, "×")

        if not self._history:
            return

        y = 16

        for text, colour, kind in self._history:
            if kind == "user":
                # Prompt echo: dim, lowercase-style quote
                p.setPen(DIM)
                p.setFont(self._font_small)
                display = text.strip('"').strip("'")
                if len(display) > 58:
                    display = display[:55] + "…"
                p.drawText(
                    QRect(pad, y, w - pad * 2 - 24, 20),
                    int(Qt.AlignmentFlag.AlignLeft) | int(Qt.AlignmentFlag.AlignVCenter),
                    display,
                )
                y += 24
                p.setPen(QPen(DIVIDER_C, 0.8))
                p.drawLine(QPoint(pad, y), QPoint(w - pad, y))
                y += 12
            else:
                lines = max(1, (len(text) + 46) // 47)
                th = lines * 24
                p.setPen(colour)
                p.setFont(self._font_body)
                _flags = int(Qt.TextFlag.TextWordWrap) | int(Qt.AlignmentFlag.AlignLeft)
                p.drawText(QRect(pad, y, w - pad * 2, th), _flags, text)
                y += th + 8

        # Clickable citation chips — painted as pill-shaped links after the
        # main body; each chip's on-screen rect is stashed for hit-testing
        # in mousePressEvent.
        if self._citations:
            self._citation_hit_rects = _paint_citation_chips(
                p,
                citations=self._citations,
                font=self._font_small,
                pad_left=pad,
                y=y,
                max_width=w - pad * 2,
            )
            y += CITATION_ROW_H + 6
        else:
            self._citation_hit_rects = []

    # ── Timers ────────────────────────────────────────────────────────────────

    def _pulse_tick(self) -> None:
        # Drive continuous repaint; breathing/bar animation reads time.monotonic()
        # directly inside paintEvent so frames are always fresh.
        if self.isVisible():
            self._pulse_on = not self._pulse_on  # kept for meeting-mode cursor blink
            self.update()

    def _do_hide(self) -> None:
        self._pulse_timer.stop()
        self._autohide_timer.stop()
        self._recording = False
        self._prompt_armed = False
        self._wake_capture_fn = None
        self.hide()

    def _present(self) -> None:
        """
        Show overlay above the current app/space without activating a new space.
        """
        self.show()
        self.raise_()
        self._reassert_window_level()

    def _reassert_window_level(self) -> None:
        """
        Re-apply NSWindow level / collection / hides-on-deactivate every
        time we show. Qt can reset these after a resize or re-parent.
        """
        try:
            ns_win = getattr(self, "_ns_window", None)
            if ns_win is None:
                return
            ns_win.setLevel_(101)  # NSPopUpMenuWindowLevel
            ns_win.setCollectionBehavior_(1 | 8 | 64 | 256)
            try:
                ns_win.setHidesOnDeactivate_(False)
            except Exception:
                pass
            ns_win.orderFrontRegardless()
        except Exception:
            pass

    def closeEvent(self, _event) -> None:  # type: ignore[override]
        self._cam_running = False  # signal camera thread to stop before Qt tears down
