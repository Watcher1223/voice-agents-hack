"""Right-edge tasks panel.

Shows the ambient-mode tasks store as a stack of cards, each with an
Approve / Dismiss button. Frameless, always on top, doesn't steal
focus — same window flags as the main overlay pill.

Thread model: the panel itself runs on the Qt main thread. Background
threads push a refresh signal via the `refresh()` method which is safe
to call from any thread; the panel polls the flag at 4 Hz and repaints
when the store changed.
"""
from __future__ import annotations

import queue
from typing import Callable

from PySide6.QtCore import Qt, QTimer, QRect  # pyright: ignore[reportMissingImports]
from PySide6.QtGui import (  # pyright: ignore[reportMissingImports]
    QColor, QFont, QGuiApplication, QPainter,
)
from PySide6.QtWidgets import QApplication, QWidget  # pyright: ignore[reportMissingImports]


# ── palette (matches main overlay) ──────────────────────────────────────────
BG      = QColor(20, 20, 24, 220)
CARD_BG = QColor(30, 30, 36, 240)
FG      = QColor(246, 246, 250)
DIM     = QColor(196, 194, 202)
FAINT   = QColor(160, 160, 170)
GREEN   = QColor(52, 199, 89)
RED     = QColor(255, 95, 87)
YELLOW  = QColor(243, 200, 75)

# ── geometry ────────────────────────────────────────────────────────────────
PANEL_W    = 320
MARGIN     = 12
HEADER_H   = 36
CARD_H     = 92
CARD_PAD   = 12
CARD_GAP   = 8
BTN_W      = 46
BTN_H      = 24
BTN_GAP    = 6
EMPTY_H    = 80


def _slot_preview(kind: str, slots: dict) -> str:
    if kind in ("compose_mail", "send_email"):
        to = str(slots.get("to", "") or "").strip()
        subj = str(slots.get("subject", "") or "").strip()
        atts = slots.get("attachments") or []
        att_note = f"  ⌘{len(atts)}" if atts else ""
        return f"→ {to or '(no recipient)':.40}  ·  {subj or '(no subject)':.36}{att_note}"
    if kind in ("send_imessage", "send_message"):
        contact = str(slots.get("contact", "") or "").strip()
        body = str(slots.get("body", "") or "").strip()
        return f"→ {contact or '(no contact)':.34}  ·  “{body:.40}”"
    if kind in ("create_calendar_event", "add_calendar_event"):
        title = str(slots.get("title", "") or "").strip()
        date = str(slots.get("date", "") or "").strip()
        time_ = str(slots.get("time", "") or "").strip()
        when = f"{date} {time_}".strip() or "(no time)"
        return f"{title:.36}  ·  {when}"
    if kind == "find_file":
        return f"query: “{str(slots.get('file_query', ''))[:40]}”"
    if kind == "open_url":
        return str(slots.get("url", ""))[:60]
    return kind


class TasksPanel(QWidget):
    """Right-edge stack of task cards.

    Callbacks:
      on_approve(task_id: str)
      on_dismiss(task_id: str)
    """

    def __init__(
        self,
        app: QApplication,
        store,
        on_approve: Callable[[str], None],
        on_dismiss: Callable[[str], None],
    ) -> None:
        super().__init__()
        self._app = app
        self._store = store
        self._on_approve = on_approve
        self._on_dismiss = on_dismiss
        # rects for hit-testing: list of (rect, task_id, action)
        self._hit_rects: list[tuple[QRect, str, str]] = []
        # refresh signal queue — safe from any thread
        self._refresh_q: queue.Queue = queue.Queue()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start()

        self._layout()
        self.show()

    # ── external API ────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Thread-safe: request a repaint at the next poll tick."""
        self._refresh_q.put(True)

    # ── lifecycle ───────────────────────────────────────────────────────

    def _poll(self) -> None:
        dirty = False
        try:
            while True:
                self._refresh_q.get_nowait()
                dirty = True
        except queue.Empty:
            pass
        if dirty:
            self._layout()
            self.update()

    def _layout(self) -> None:
        screen = QGuiApplication.primaryScreen().availableGeometry()
        pending = self._store.pending()
        if pending:
            body_h = HEADER_H + CARD_PAD + len(pending) * (CARD_H + CARD_GAP) + CARD_PAD
        else:
            body_h = HEADER_H + EMPTY_H
        body_h = min(body_h, screen.height() - MARGIN * 2)
        x = screen.x() + screen.width() - PANEL_W - MARGIN
        y = screen.y() + MARGIN + 40       # small gap below menu bar / top pill
        self.setGeometry(x, y, PANEL_W, body_h)

    # ── paint ───────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
        )
        r = self.rect()
        p.setBrush(BG)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, 18, 18)

        # header
        p.setPen(FG)
        p.setFont(QFont(".AppleSystemUIFont", 13, QFont.Weight.Bold))
        p.drawText(
            QRect(CARD_PAD, 0, r.width() - 2 * CARD_PAD, HEADER_H),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "Tasks",
        )
        # count badge
        pending = self._store.pending()
        if pending:
            p.setPen(DIM)
            p.setFont(QFont(".AppleSystemUIFont", 11))
            p.drawText(
                QRect(CARD_PAD, 0, r.width() - 2 * CARD_PAD, HEADER_H),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{len(pending)} pending",
            )

        # empty state
        self._hit_rects.clear()
        if not pending:
            p.setPen(FAINT)
            p.setFont(QFont(".AppleSystemUIFont", 12))
            p.drawText(
                QRect(CARD_PAD, HEADER_H, r.width() - 2 * CARD_PAD, EMPTY_H),
                Qt.AlignmentFlag.AlignCenter,
                "Nothing yet — keep talking. Action\nitems will show up here.",
            )
            return

        # cards
        y = HEADER_H + CARD_PAD // 2
        for task in pending:
            card_rect = QRect(CARD_PAD, y, PANEL_W - 2 * CARD_PAD, CARD_H)
            p.setBrush(CARD_BG)
            p.drawRoundedRect(card_rect, 10, 10)

            # headline
            p.setPen(FG)
            p.setFont(QFont(".AppleSystemUIFont", 12, QFont.Weight.Medium))
            p.drawText(
                QRect(card_rect.x() + 12, card_rect.y() + 8, card_rect.width() - 24, 22),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                task.headline[:42],
            )

            # slots preview
            p.setPen(DIM)
            p.setFont(QFont(".AppleSystemUIFont", 11))
            p.drawText(
                QRect(card_rect.x() + 12, card_rect.y() + 30, card_rect.width() - 24, 20),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                _slot_preview(task.action_text, task.slots),
            )

            # buttons
            btn_y = card_rect.y() + card_rect.height() - BTN_H - 10
            approve = QRect(
                card_rect.x() + card_rect.width() - (BTN_W * 2 + BTN_GAP) - 12,
                btn_y, BTN_W, BTN_H,
            )
            dismiss = QRect(
                card_rect.x() + card_rect.width() - BTN_W - 12,
                btn_y, BTN_W, BTN_H,
            )
            p.setBrush(GREEN)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(approve, 6, 6)
            p.setPen(Qt.GlobalColor.white)
            p.setFont(QFont(".AppleSystemUIFont", 12, QFont.Weight.Bold))
            p.drawText(approve, Qt.AlignmentFlag.AlignCenter, "✓")
            p.setBrush(QColor(50, 50, 58, 240))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(dismiss, 6, 6)
            p.setPen(DIM)
            p.drawText(dismiss, Qt.AlignmentFlag.AlignCenter, "✗")

            self._hit_rects.append((approve, task.id, "approve"))
            self._hit_rects.append((dismiss, task.id, "dismiss"))

            # status decoration (executing dot, etc.)
            if task.status == "executing":
                p.setPen(YELLOW)
                p.setFont(QFont(".AppleSystemUIFont", 10))
                p.drawText(
                    QRect(card_rect.x() + 12, card_rect.y() + card_rect.height() - 22,
                          card_rect.width() - 24 - (BTN_W * 2 + BTN_GAP + 12), 14),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    "▶ running…",
                )

            y += CARD_H + CARD_GAP

    # ── input ───────────────────────────────────────────────────────────

    def mousePressEvent(self, e) -> None:  # type: ignore[override]
        if e.button() != Qt.MouseButton.LeftButton:
            return
        x, y = e.position().x(), e.position().y()
        for rect, tid, action in self._hit_rects:
            if rect.contains(int(x), int(y)):
                try:
                    if action == "approve":
                        self._on_approve(tid)
                    else:
                        self._on_dismiss(tid)
                except Exception:
                    pass
                self.refresh()
                return
