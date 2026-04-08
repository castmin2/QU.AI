# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbox (Futura Minimal)
- Matches System Monitor aesthetic (light/dark, rounded cards, Segoe UI)
- Safety badge (reads latest ai_safety_report_*.json)
- Non-agentic guardrails + Safe Mode gate via QUANTONIUM_SAFE_MODE
- Message bubbles, typing indicator, transcript logging
- Hooks to wire your responder (weights/organized/*.json) later
"""

import os, sys, json, glob, time, datetime, re, logging
import traceback
from typing import Optional, Dict, Any, List
from pathlib import Path

import threading
import subprocess
import argparse

# Ensure local project imports work for both:
# - python -m src.apps.qshll_chatbox
# - python src/apps/qshll_chatbox.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from atomic_io import AtomicJsonlWriter, atomic_write_text
from src.apps.local_agent_tools import list_files, read_file, search_repo
from src.apps.ai_model_wrapper import discover_local_models, set_model_id
from src.apps.local_runtime_profiles import apply_profile_env, profile_for_model
from src.apps.rftmw_pack_discovery import discover_rftmw_packs

from PyQt5.QtCore import Qt, QTimer, QSize, QPoint, QEvent, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush, QTextOption
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QScrollArea, QFrame, QFileDialog,
    QStatusBar, QMessageBox, QComboBox
)

# Configure module-level logger once for reuse
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _format_model_label(model_id: str, *, backend: str, detail: str = "", chat_capable: bool = True) -> str:
    detail = (detail or "").strip()
    capability = "" if chat_capable else " | embedding-only"
    if detail:
        return f"{model_id} [{detail}] ({backend}{capability})"
    return f"{model_id} ({backend}{capability})"


def _log_crash(context: str, exc: BaseException, tb_obj=None) -> None:
    os.makedirs("logs", exist_ok=True)
    crash_path = os.path.join("logs", "chatbox_crash.log")
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [
        f"[{ts}] context={context}",
        f"type={type(exc).__name__}",
        f"message={exc}",
    ]
    if tb_obj is not None:
        lines.append("traceback:")
        lines.extend(traceback.format_tb(tb_obj))
    else:
        lines.append("traceback:")
        lines.extend(traceback.format_exception(type(exc), exc, exc.__traceback__))
    with open(crash_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n\n")


def _install_crash_hooks() -> None:
    def _main_hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            return
        _log_crash("main-thread", exc_value, exc_tb)
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    def _thread_hook(args):
        _log_crash(f"thread:{getattr(args, 'thread', None)}", args.exc_value, args.exc_traceback)

    sys.excepthook = _main_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook

DEFAULT_OLLAMA_CHAT_MODEL = "qwen2.5-coder:3b"
ENABLE_LEGACY_QUANTUM = os.getenv("QUANTONIUM_ENABLE_LEGACY_QUANTUM", "0") == "1"

# Legacy quantum stack is opt-in now. The main chatbox is local-first.
if ENABLE_LEGACY_QUANTUM:
    try:
        from quantum_safety_system import QuantumSafetySystem
        from quantum_conversation_manager import QuantumConversationManager
        from quantum_rlhf_system import QuantumRLHFSystem
        from quantum_domain_fine_tuner import QuantumDomainFineTuner
        QUANTUM_AI_AVAILABLE = True
        logger.debug("Legacy Quantum AI System enabled")
    except ImportError as e:
        QUANTUM_AI_AVAILABLE = False
        logger.warning("Legacy Quantum AI System not available: %s", e)
else:
    QUANTUM_AI_AVAILABLE = False

# ---------- Reusable UI primitives (mirrors System Monitor) ----------
def load_latest_safety_report_text() -> Optional[str]:
    files = sorted(glob.glob("ai_safety_report_*.json"), key=os.path.getmtime, reverse=True)
    if not files: return None
    try:
        return open(files[0], "r", encoding="utf-8").read()
    except Exception:
        return None

def safety_is_green() -> tuple[bool, Optional[str]]:
    txt = load_latest_safety_report_text()
    if not txt: return (False, None)
    return ("FAIL Non Agentic Constraints" not in txt, txt)

def latest_report_time() -> Optional[str]:
    files = sorted(glob.glob("ai_safety_report_*.json"), key=os.path.getmtime, reverse=True)
    if not files: return None
    ts = os.path.getmtime(files[0])
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _torch_import_healthcheck(*, timeout_s: float = 8.0) -> tuple[bool, str]:
    """
    Best-effort check for whether importing torch is safe in this environment.

    Important: some Windows torch DLL issues can terminate the interpreter.
    This check runs the import in a subprocess so the GUI process stays alive.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import torch; print(getattr(torch, '__version__', 'ok'))"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode == 0:
            return True, (proc.stdout or "").strip()
        msg = (proc.stderr or proc.stdout or "").strip()
        return False, msg or f"torch import failed (exit {proc.returncode})"
    except subprocess.TimeoutExpired:
        return False, "torch import timed out"
    except Exception as e:
        return False, str(e)


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of a single JSON object from model text."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, min(len(text), start + 20000)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                try:
                    obj = json.loads(chunk)
                except Exception:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


class ReplyWorker(QObject):
    finished = pyqtSignal(str, str, float)
    failed = pyqtSignal(str, str)

    def __init__(self, chatbox: "Chatbox", prompt: str):
        super().__init__()
        self._chatbox = chatbox
        self._prompt = prompt

    def run(self) -> None:
        try:
            reply, conf = self._chatbox._generate_reply_sync(self._prompt)
            self.finished.emit(self._prompt, reply, conf)
        except Exception as e:
            self.failed.emit(self._prompt, f"{type(e).__name__}: {e}")

class Card(QFrame):
    def __init__(self, title: str = ""):
        super().__init__()
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(8)
        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("CardTitle")
        lay.addWidget(self.title_lbl)
        self.body = QWidget()
        lay.addWidget(self.body)

# ---------- Message bubble ----------
class Bubble(QWidget):
    def __init__(self, text: str, me: bool, light: bool):
        super().__init__()
        self.text = text
        self.me = me
        self.light = light
        self.setMinimumHeight(40)
        self.setSizePolicy(self.sizePolicy().Expanding, self.sizePolicy().Minimum)
        # Set a reasonable maximum width to prevent text from being too wide
        self.setMaximumWidth(480)

    def sizeHint(self) -> QSize:
        # Calculate proper size based on text wrapping
        fm = self.fontMetrics()
        available_width = min(480, self.parent().width() - 50 if self.parent() else 480)
        text_margin = 12
        pad = 10
        
        # Break text into lines
        words = self.text.split(" ")
        lines = []
        current_line = ""
        
        for word in words:
            test_line = (current_line + " " + word).strip()
            if fm.width(test_line) > available_width - 2*text_margin and current_line:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
        
        if not lines:
            lines = [""]
        
        # Calculate height
        line_height = fm.height()
        total_height = len(lines) * line_height + 2*pad + 12  # Extra padding for widget
        
        return QSize(available_width, max(40, total_height))

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect().adjusted(8, 6, -8, -6)

        # Calculate available width based on widget size
        available_width = min(r.width(), 480)
        text_margin = 12
        pad = 10
        
        # choose colors
        if self.me:
            bg = QColor(80, 180, 120, 255) if self.light else QColor(38, 142, 96, 255)
            fg = QColor(255, 255, 255)
        else:
            bg = QColor(230, 238, 246, 255) if self.light else QColor(29, 43, 58, 255)
            fg = QColor(36, 51, 66) if self.light else QColor(223, 231, 239)

        # improved text wrapping
        fm = self.fontMetrics()
        words = self.text.split(" ")
        lines = []
        current_line = ""
        
        for word in words:
            test_line = (current_line + " " + word).strip()
            test_width = fm.width(test_line)
            
            if test_width > available_width - 2*text_margin and current_line:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
        
        # ensure we have at least one line
        if not lines:
            lines = [""]
        
        # calculate bubble dimensions
        line_height = fm.height()
        text_h = len(lines) * line_height
        max_line_width = max(fm.width(line) for line in lines) if lines else 100
        bw = min(available_width, max_line_width + 2*text_margin)
        bh = text_h + 2*pad

        # position bubble
        if self.me:
            bx = r.right() - bw
        else:
            bx = r.left()
        by = r.top()

        # draw bubble background
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(bx, by, bw, bh, 10, 10)

        # draw text line by line
        p.setPen(fg)
        tx = bx + text_margin
        ty = by + pad + fm.ascent()
        
        for i, line in enumerate(lines):
            p.drawText(tx, ty + i*line_height, line)
        
        p.end()

# ---------- Chatbox main ----------
class Chatbox(QMainWindow):
    def __init__(self):
        super().__init__()
        print("Chatbox constructor started...")
        self.setWindowTitle("Chatbox")
        self.resize(980, 720)
        self._light = True
        self._safe_mode = (os.getenv("QUANTONIUM_SAFE_MODE") == "1")
        self._log_fp = None
        self._ensure_logfile()
        
        print("Basic setup complete, initializing local chat runtime...")
        # Legacy quantum stack is optional; Ollama local chat is the main path.
        if QUANTUM_AI_AVAILABLE:
            try:
                self._safety_system = QuantumSafetySystem()
                self._conversation_manager = QuantumConversationManager()
                self._rlhf_system = QuantumRLHFSystem()
                self._conversation_id = self._conversation_manager.start_conversation()
                self._quantum_ai_enabled = True
                # Startup log for encoded backend presence
                try:
                    enc = getattr(self._conversation_manager.inference_engine, 'encoded_backend', None)
                    with open('logs/startup.log', 'a', encoding='utf-8') as sf:
                        sf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Encoded backend loaded: {enc is not None}\n")
                except Exception:
                    pass
                print("AI system initialized (Safety + RLHF + Memory + Domains)")
            except Exception as e:
                print(f"AI system initialization failed: {e}")
                self._quantum_ai_enabled = False
        else:
            self._quantum_ai_enabled = False
            print("Legacy Quantum AI disabled; using local Ollama chat")

        # Set legacy attributes for UI compatibility
        self._learning_enabled = self._quantum_ai_enabled
        self._quantum_enabled = self._quantum_ai_enabled

        # Local LLM main path (on-device).
        # Default: enabled unless explicitly disabled.
        # Override:
        # - QUANTONIUM_LOCAL_LLM=0 disables local LLM entirely.
        env_llm = os.getenv("QUANTONIUM_LOCAL_LLM")
        if env_llm == "0":
            self._local_llm_enabled = False
        else:
            self._local_llm_enabled = True
        # Local backend selector (when local LLM is enabled)
        # - QUANTONIUM_LOCAL_BACKEND=ollama|hf|rftmw (default: ollama)
        self._local_backend = os.getenv("QUANTONIUM_LOCAL_BACKEND", "ollama").strip().lower()
        if not os.getenv("QUANTONIUM_OLLAMA_MODEL", "").strip():
            os.environ["QUANTONIUM_OLLAMA_MODEL"] = DEFAULT_OLLAMA_CHAT_MODEL
        self._torch_ok = True
        self._local_llm_disabled_reason = ""
        if self._local_llm_enabled and self._local_backend in {"hf", "rftmw"} and os.getenv("QUANTONIUM_TORCH_HEALTHCHECK", "1") != "0":
            ok, detail = _torch_import_healthcheck()
            self._torch_ok = bool(ok)
            if not ok:
                self._local_llm_enabled = False
                self._local_llm_disabled_reason = (
                    f"Local LLM disabled: backend {self._local_backend} requires torch, and torch "
                    f"failed to import in a subprocess. Details: {detail}"
                )
                print(self._local_llm_disabled_reason)
        self._chat_history: List[tuple[str, str]] = []
        self._reply_thread: Optional[QThread] = None
        self._reply_worker: Optional[ReplyWorker] = None
        self._reply_in_flight = False
        self._rftmw_engine = None
        self._rftmw_provenance: Dict[str, Any] = {}
        self._rftmw_pack_path = os.getenv("QUANTONIUM_RFTMW_CACHE_PATH", "").strip()
        self._rftmw_loaded_pack_path = ""
        self._last_topology_context: Dict[str, Any] = {}
        self._local_profile: Dict[str, Any] = {}
        self._model_entries: List[Dict[str, Any]] = []
        self._current_model_entry: Dict[str, Any] = {}
        self._rftmw_model_id = os.getenv(
            "QUANTONIUM_RFTMW_MODEL_ID",
            os.getenv("QUANTONIUM_MODEL_ID", ""),
        )
        self._prime_local_model_selection()
        initial_model = (
            os.getenv("QUANTONIUM_OLLAMA_MODEL", "").strip()
            if self._local_backend == "ollama"
            else self._rftmw_model_id.strip()
        )
        if initial_model:
            try:
                initial_profile = profile_for_model(initial_model, self._local_backend)
                apply_profile_env(initial_profile)
                self._local_profile = initial_profile.as_dict()
            except Exception:
                self._local_profile = {}
        self._rftmw_layer_limit = _env_int("QUANTONIUM_RFTMW_LAYER_LIMIT", None)
        self._agent_mode = (os.getenv("QUANTONIUM_AGENT_MODE", "0") == "1")
        self._agent_max_steps = _env_int("QUANTONIUM_AGENT_MAX_STEPS", 4) or 4

        print("Trainer initialized, building UI...")
        self._build_ui()
        if self._local_llm_disabled_reason:
            try:
                self.statusBar().showMessage(self._local_llm_disabled_reason, 12000)
            except Exception:
                pass
        print("UI built, applying styles...")
        self._apply_style(light=True)
        print("Styles applied, refreshing safety badge...")
        self._refresh_safety_badge()  # initial
        print("Chatbox fully initialized")
        self._badge_timer = QTimer(self); self._badge_timer.timeout.connect(self._refresh_safety_badge); self._badge_timer.start(2000)

        if self._safe_mode:
            self._disable_input_for_safe_mode()

    def _discover_chatbox_models(self) -> list[Dict[str, Any]]:
        items: list[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for pack in discover_rftmw_packs():
            pack_label_id = pack.display_name or pack.model_name or Path(pack.path).stem
            pack_hash = f" sha:{pack.state_dict_hash_sha256[:10]}..." if pack.state_dict_hash_sha256 else ""
            pack_detail = f"pack {Path(pack.path).name}{pack_hash}".strip()
            runtime_model_id = str(pack.runtime_model_id or "").strip()
            items.append(
                {
                    "label": _format_model_label(
                        pack_label_id,
                        backend="rftmw",
                        detail=pack_detail,
                        chat_capable=bool(runtime_model_id),
                    ),
                    "model_id": runtime_model_id or pack_label_id,
                    "backend": "rftmw",
                    "chat_capable": bool(runtime_model_id),
                    "source": pack.source,
                    "cache_path": pack.path,
                    "display_name": pack.display_name,
                    "original_model_name": pack.model_name,
                }
            )
            seen.add(("rftmw_pack", pack.path.lower()))

        try:
            from src.apps.ollama_client import discover_ollama_models

            for candidate in discover_ollama_models(include_embeddings=True):
                key = ("ollama", candidate.model_id)
                if key in seen:
                    continue
                seen.add(key)
                detail_parts = [candidate.parameter_size, candidate.quantization_level]
                items.append(
                    {
                        "label": _format_model_label(
                            candidate.model_id,
                            backend="ollama",
                            detail=" ".join(part for part in detail_parts if part and part != "?"),
                            chat_capable=not candidate.is_embedding,
                        ),
                        "model_id": candidate.model_id,
                        "backend": "ollama",
                        "chat_capable": not candidate.is_embedding,
                        "source": "ollama",
                    }
                )
        except Exception as e:
            print(f"Model discovery warning (ollama): {e}")

        for candidate in discover_local_models():
            backend = "ollama" if candidate.source == "ollama" else "rftmw"
            key = (backend, candidate.model_id)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "label": _format_model_label(
                        candidate.model_id,
                        backend=backend,
                        detail=candidate.detail,
                    ),
                    "model_id": candidate.model_id,
                    "backend": backend,
                    "chat_capable": True,
                    "source": candidate.source,
                }
            )

        return items

    def _prime_local_model_selection(self) -> None:
        self._model_entries = self._discover_chatbox_models()

        current_backend = self._local_backend if self._local_backend in {"ollama", "rftmw"} else "rftmw"
        current_cache_path = os.getenv("QUANTONIUM_RFTMW_CACHE_PATH", "").strip().lower()
        current_model = (
            os.getenv("QUANTONIUM_OLLAMA_MODEL", "").strip()
            if current_backend == "ollama"
            else self._rftmw_model_id.strip()
        )

        selected = None
        for entry in self._model_entries:
            entry_cache_path = str(entry.get("cache_path", "")).strip().lower()
            if current_backend == "rftmw" and current_cache_path and entry_cache_path == current_cache_path:
                selected = entry
                break
            if entry["backend"] == current_backend and entry["model_id"] == current_model:
                selected = entry
                break

        if selected is None:
            for entry in self._model_entries:
                if entry["backend"] == "rftmw" and bool(entry.get("cache_path")) and entry.get("chat_capable", True):
                    selected = entry
                    break

        if selected is None:
            for entry in self._model_entries:
                if entry["backend"] == "ollama" and entry["model_id"] == DEFAULT_OLLAMA_CHAT_MODEL:
                    selected = entry
                    break

        if selected is None:
            for entry in self._model_entries:
                if entry["backend"] == "ollama" and entry.get("chat_capable", True):
                    selected = entry
                    break

        if selected is None:
            for entry in self._model_entries:
                if entry.get("chat_capable", True):
                    selected = entry
                    break

        if selected is None and self._model_entries:
            selected = self._model_entries[0]

        if selected is not None:
            self._apply_selected_model(
                selected["backend"],
                selected["model_id"],
                entry=selected,
                refresh_combo=False,
            )

    def _augment_system_prompt_with_topology(self, system_prompt: str, user_prompt: str) -> str:
        if os.getenv("QUANTONIUM_TOPOLOGICAL_SPACE", "1") == "0":
            self._last_topology_context = {}
            return system_prompt

        try:
            from src.apps.topological_chat_space import (
                build_topological_chat_context,
                render_topological_context_block,
            )

            context = build_topological_chat_context(user_prompt)
            self._last_topology_context = context.as_dict()
            return (
                system_prompt
                + "\n\n"
                + "Use the following deterministic topological/RFT context as auxiliary "
                + "whitebox state derived from the user's prompt:\n"
                + render_topological_context_block(context)
            )
        except Exception as e:
            self._last_topology_context = {"error": f"{type(e).__name__}: {e}"}
            print(f"Topological prompt augmentation failed: {e}")
            return system_prompt

    def _get_all_available_models(self) -> list[Dict[str, Any]]:
        """Get all available local models with their real backend wiring."""
        items = self._discover_chatbox_models()
        if not items:
            return [
                {
                    "label": "No models found - run ollama pull qwen2.5-coder:3b",
                    "model_id": "",
                    "backend": "",
                    "chat_capable": False,
                    "source": "",
                }
            ]
        self._model_entries = items
        return items

    def _apply_selected_model(self, backend: str, model_id: str, *, entry: Optional[Dict[str, Any]] = None, refresh_combo: bool = True) -> None:
        model_id = (model_id or "").strip()
        if not model_id:
            return

        backend = (backend or "").strip().lower()
        selected_entry = dict(entry or {})
        cache_path = str(selected_entry.get("cache_path", "")).strip()
        profile = profile_for_model(model_id, backend)
        apply_profile_env(profile)
        self._local_profile = profile.as_dict()
        self._local_backend = backend
        self._current_model_entry = selected_entry
        os.environ["QUANTONIUM_LOCAL_BACKEND"] = backend
        if backend == "ollama":
            os.environ["QUANTONIUM_OLLAMA_MODEL"] = model_id
            os.environ.pop("QUANTONIUM_RFTMW_CACHE_PATH", None)
            self._rftmw_pack_path = ""
            self._rftmw_loaded_pack_path = ""
        elif backend == "rftmw":
            os.environ["QUANTONIUM_RFTMW_MODEL_ID"] = model_id
            self._rftmw_model_id = model_id
            self._rftmw_engine = None
            self._rftmw_provenance = {}
            self._rftmw_loaded_pack_path = ""
            self._rftmw_pack_path = cache_path
            if cache_path:
                os.environ["QUANTONIUM_RFTMW_CACHE_PATH"] = cache_path
            else:
                os.environ.pop("QUANTONIUM_RFTMW_CACHE_PATH", None)
            set_model_id(model_id)
        if refresh_combo:
            self._refresh_model_combo()

    def _refresh_model_combo(self) -> None:
        if getattr(self, "model_combo", None) is None:
            return

        items = self._get_all_available_models()
        current_backend = getattr(self, "_local_backend", "").strip().lower()
        current_cache_path = os.getenv("QUANTONIUM_RFTMW_CACHE_PATH", "").strip().lower()
        current_model = (
            os.getenv("QUANTONIUM_OLLAMA_MODEL", "").strip().lower()
            if current_backend == "ollama"
            else self._rftmw_model_id.strip().lower()
        )

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for item in items:
            self.model_combo.addItem(item["label"], item)

        if current_backend and current_model:
            for i in range(self.model_combo.count()):
                entry = self.model_combo.itemData(i) or {}
                entry_cache_path = str(entry.get("cache_path", "")).strip().lower()
                if current_backend == "rftmw" and current_cache_path and entry_cache_path == current_cache_path:
                    self.model_combo.setCurrentIndex(i)
                    break
                if (
                    str(entry.get("backend", "")).strip().lower() == current_backend
                    and str(entry.get("model_id", "")).strip().lower() == current_model
                ):
                    self.model_combo.setCurrentIndex(i)
                    break
        self.model_combo.setEnabled(any(str((self.model_combo.itemData(i) or {}).get("model_id", "")).strip() for i in range(self.model_combo.count())))
        self.model_combo.blockSignals(False)



    def _on_model_changed(self, _idx: int) -> None:
        if getattr(self, "model_combo", None) is None:
            return

        entry = self.model_combo.currentData() or {}
        model_id = str(entry.get("model_id", "")).strip()
        if not model_id:
            return

        self._apply_selected_model(str(entry.get("backend", "rftmw")), model_id, entry=entry, refresh_combo=False)
        self._refresh_safety_badge()
        if not bool(entry.get("chat_capable", True)):
            self.statusBar().showMessage(f"Model set: {model_id} (embedding-only; chat disabled)", 6000)
        else:
            self.statusBar().showMessage(f"Model set: {entry.get('backend', 'rftmw')}:{model_id}", 5000)

    # ---------- UI ----------
    def _build_ui(self):
        print("Starting UI construction...")
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        print("Central widget created...")

        # header
        header = QWidget(); header.setFixedHeight(60)
        hl = QVBoxLayout(header); hl.setContentsMargins(20,8,20,8)
        title = QLabel("Chatbox"); title.setObjectName("Title")
        subtitle = QLabel("Reactive assistant - Non-agentic")
        subtitle.setObjectName("SubTitle")
        hl.addWidget(title); hl.addWidget(subtitle)
        root.addWidget(header)
        print("Header created...")

        # controls row
        ctrl = QWidget(); cl = QHBoxLayout(ctrl); cl.setContentsMargins(16, 8, 16, 8)
        self.theme_btn = QPushButton("Dark/Light"); self.theme_btn.clicked.connect(self.toggle_theme)
        self.clear_btn = QPushButton("Clear"); self.clear_btn.clicked.connect(self.clear_chat)
        self.save_btn = QPushButton("Save Transcript"); self.save_btn.clicked.connect(self.save_transcript)
        print("Control buttons created...")

        # Local model selector
        self.model_combo = None
        try:
            if getattr(self, "_local_llm_enabled", False):
                backend_label = QLabel("Model")
                backend_label.setStyleSheet("font-weight: bold;")
                
                self.model_combo = QComboBox()
                self.model_combo.setMinimumWidth(360)
                self._refresh_model_combo()
                self.model_combo.currentIndexChanged.connect(self._on_model_changed)

                cl.addWidget(backend_label)
                cl.addWidget(self.model_combo)
        except Exception:
            self.model_combo = None
        
        # Add training controls if available
        if self._learning_enabled:
            self.train_btn = QPushButton("Training Stats"); self.train_btn.clicked.connect(self.show_training_stats)
            cl.addWidget(self.train_btn)
            print("Training controls added...")
        
        cl.addWidget(self.theme_btn); cl.addWidget(self.clear_btn); cl.addWidget(self.save_btn)
        cl.addStretch(1)
        print("Controls layout complete...")
        # safety badge
        self.safety_badge = QLabel("..."); self.safety_badge.setObjectName("Badge")
        cl.addWidget(self.safety_badge)
        root.addWidget(ctrl)
        print("Safety badge added...")

        # main area: left info card + chat scroll
        main = QWidget(); ml = QHBoxLayout(main); ml.setContentsMargins(16, 8, 16, 8); ml.setSpacing(16)
        print("Main area created...")

        # left: info card
        self.info_card = Card("Session")
        il = QVBoxLayout(self.info_card.body); il.setSpacing(6)
        self.info_text = QLabel("Mode: Reactive (non-agentic)\nSafety: --\nTranscript: active")
        il.addWidget(self.info_text)
        il.addStretch(1)
        ml.addWidget(self.info_card)
        print("Info card created...")

        # chat area (scroll)
        self.chat_card = Card("Conversation")
        chat_body = QVBoxLayout(self.chat_card.body); chat_body.setContentsMargins(0,0,0,0)
        self.scroll = QScrollArea(); 
        self.scroll.setWidgetResizable(True); 
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Prevent horizontal scrolling
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        print("Chat scroll area created...")
        
        self.scroll_wrap = QWidget(); 
        self.scroll_v = QVBoxLayout(self.scroll_wrap); 
        self.scroll_v.setContentsMargins(12,12,12,12); 
        self.scroll_v.setSpacing(10)
        self.scroll_v.addStretch(1)
        self.scroll.setWidget(self.scroll_wrap)
        chat_body.addWidget(self.scroll)
        ml.addWidget(self.chat_card, 1)
        root.addWidget(main, 1)

        # input row
        inrow = QWidget()
        ir = QHBoxLayout(inrow)
        ir.setContentsMargins(16, 8, 16, 16)
        ir.setSpacing(8)
        self.input = QTextEdit()
        self.input.setFixedHeight(80)
        self.input.setPlaceholderText("Type a message (Enter = send, Shift+Enter = new line)")
        self.input.installEventFilter(self)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        # feedback buttons (persisted to logs/feedback.jsonl)
        self.up_btn = QPushButton("Up")
        self.up_btn.clicked.connect(lambda: self._persist_feedback(True))
        self.down_btn = QPushButton("Down")
        self.down_btn.clicked.connect(lambda: self._persist_feedback(False))
        ir.addWidget(self.input, 1)
        ir.addWidget(self.send_btn)
        ir.addWidget(self.up_btn)
        ir.addWidget(self.down_btn)
        root.addWidget(inrow)

        self.setStatusBar(QStatusBar())

    def _apply_style(self, light=True):
        self._light = light
        if light:
            qss = """
            QMainWindow, QWidget { background:#fafafa; color:#243342; font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif; }
            #Title { font-size:20px; font-weight:300; color:#2c3e50; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            QFrame#Card { border:1px solid #e9ecef; border-radius:14px; background:#ffffff; }
            QLabel#CardTitle { color:#6c7f90; font-size:12px; letter-spacing:.4px; }
            QPushButton { background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; padding:8px 14px; color:#495057; }
            QPushButton:hover { background:#eef2f6; }
            QTextEdit { background:#ffffff; border:1px solid #e9ecef; border-radius:8px; padding:8px 10px; }
            QLabel#Badge { background:#e3f2fd; color:#1976d2; border:1px solid #b6d9ff; border-radius:10px; padding:6px 10px; }
            """
        else:
            qss = """
            QMainWindow, QWidget { background:#0f1216; color:#dfe7ef; font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif; }
            #Title { font-size:20px; font-weight:300; color:#dfe7ef; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            QFrame#Card { border:1px solid #1f2a36; border-radius:14px; background:#12161b; }
            QLabel#CardTitle { color:#8aa0b3; font-size:12px; letter-spacing:.4px; }
            QPushButton { background:#12161b; border:1px solid #2a3847; border-radius:8px; padding:8px 14px; color:#c8d3de; }
            QPushButton:hover { background:#17202a; }
            QTextEdit { background:#12161b; border:1px solid #1f2a36; border-radius:8px; padding:8px 10px; color:#e8eff7; }
            QLabel#Badge { background:#1d2b3a; color:#7dc4ff; border:1px solid #2a3a4a; border-radius:10px; padding:6px 10px; }
            """
        self.setStyleSheet(qss)
        self.repaint()

    def toggle_theme(self):
        self._apply_style(not self._light)

    # ---------- Safety badge ----------
    def _disable_input_for_safe_mode(self):
        self.input.setDisabled(True)
        self.send_btn.setDisabled(True)
        self.input.setPlaceholderText("Safety mode is active - model responses disabled until next successful safety check.")

    def _refresh_safety_badge(self):
        ok, _txt = safety_is_green()
        when = latest_report_time() or "n/a"
        if os.getenv("QUANTONIUM_SAFE_MODE") == "1":
            self.safety_badge.setText(f"SAFE MODE | {when}")
        else:
            self.safety_badge.setText(f"{'OK' if ok else 'WARN'} Non-Agentic | {when}")

        if getattr(self, "_local_llm_enabled", False):
            backend = getattr(self, "_local_backend", "hf")
            topology_tag = " [topology]" if os.getenv("QUANTONIUM_TOPOLOGICAL_SPACE", "1") != "0" else ""
            if backend == "ollama":
                local_model = os.getenv("QUANTONIUM_OLLAMA_MODEL", DEFAULT_OLLAMA_CHAT_MODEL)
                local_line = f"Local LLM: ollama:{local_model}{topology_tag}"
            elif backend == "rftmw":
                local_line = f"Local LLM: rftmw:{self._rftmw_model_id}"
                state_hash = self._rftmw_provenance.get("state_dict_hash_sha256", "")
                if state_hash:
                    local_line += f" (sha256:{state_hash[:10]}...)"
                active_pack = self._rftmw_loaded_pack_path or self._rftmw_pack_path
                if active_pack:
                    local_line += f" [pack:{Path(active_pack).name}]"
                local_line += topology_tag
                if (
                    os.getenv("QUANTONIUM_RFT_FILE_MODE", "0") == "1"
                    or bool(os.getenv("QUANTONIUM_QUANTUM_WEIGHTS_JSON", "").strip())
                ):
                    local_line += " [file-mode]"
            profile_name = self._local_profile.get("profile_name", "") if isinstance(self._local_profile, dict) else ""
            if profile_name:
                local_line += f" | profile:{profile_name}"
        else:
            local_line = "Local LLM: off"

        self.info_text.setText(
            f"Mode: {'SAFE MODE' if os.getenv('QUANTONIUM_SAFE_MODE')=='1' else ('Agentic' if getattr(self, '_agent_mode', False) else 'Reactive (non-agentic)')}\n"
            f"Safety: {'verified' if ok else 'check'}\n"
            f"Transcript: active\n"
            f"{local_line}"
        )

    def _build_rftmw_prompt(self, user_prompt: str) -> str:
        # Use the same prompt formatting as the local HF backend so chat models
        # get proper ChatML structure.
        system_prompt = (
            "You are a helpful, non-agentic assistant. "
            "Refuse downloads, external network access, and system modification commands."
        )
        system_prompt = self._augment_system_prompt_with_topology(system_prompt, user_prompt)
        from src.apps.ai_model_wrapper import format_prompt_auto

        return format_prompt_auto(
            user_prompt,
            history=self._chat_history,
            system_prompt=system_prompt,
            max_turns=6,
        )

    def _load_rftmw_engine(self):
        if self._rftmw_engine is not None:
            return self._rftmw_engine

        from quantonium_os_src.engine.rftmw_inference import CompressedInferenceEngine

        entropy_threshold = _env_float("QUANTONIUM_RFTMW_ENTROPY_THRESHOLD", 0.40)
        keep_ratio = _env_float("QUANTONIUM_RFTMW_KEEP_RATIO", 0.30)
        kv_keep_ratio = _env_float("QUANTONIUM_RFTMW_KV_KEEP_RATIO", 0.30)
        max_rft_elems = _env_int("QUANTONIUM_RFTMW_MAX_RFT_ELEMS", 2_000_000)
        local_only = (os.getenv("QUANTONIUM_LOCAL_ONLY", "1") != "0")
        compress_kv = (os.getenv("QUANTONIUM_RFTMW_COMPRESS_KV", "1") != "0")

        self.statusBar().showMessage(
            f"Initializing RFTMW backend for {self._rftmw_model_id}...",
            5000,
        )
        engine = CompressedInferenceEngine(
            model_name_or_path=self._rftmw_model_id,
            entropy_threshold=entropy_threshold,
            weight_keep_ratio=keep_ratio,
            kv_keep_ratio=kv_keep_ratio,
            compress_kv=compress_kv,
            device="cpu",
            local_files_only=local_only,
            max_rft_elements=max_rft_elems,
        )
        engine.compress_model(layer_limit=self._rftmw_layer_limit, verbose=False)
        self._rftmw_engine = engine
        self._rftmw_provenance = engine.provenance()
        self._rftmw_loaded_pack_path = getattr(engine.memory, "_loaded_from_pack", "") or ""
        if self._rftmw_loaded_pack_path:
            self._rftmw_pack_path = self._rftmw_loaded_pack_path
        elif os.getenv("QUANTONIUM_RFTMW_CACHE_PATH", "").strip():
            self._rftmw_pack_path = os.getenv("QUANTONIUM_RFTMW_CACHE_PATH", "").strip()
        self._refresh_safety_badge()
        if self._rftmw_loaded_pack_path:
            self.statusBar().showMessage(
                f"RFTMW backend ready from pack: {Path(self._rftmw_loaded_pack_path).name}",
                5000,
            )
        else:
            self.statusBar().showMessage("RFTMW backend ready", 3000)
        return self._rftmw_engine

    # ---------- Chat plumbing ----------
    def eventFilter(self, obj, ev):
        if obj is self.input and ev.type() == QEvent.KeyPress:
            if ev.key() in (Qt.Key_Return, Qt.Key_Enter) and not (ev.modifiers() & Qt.ShiftModifier):
                self.send_message(); return True
        return super().eventFilter(obj, ev)

    def clear_chat(self):
        # remove bubbles
        for i in reversed(range(self.scroll_v.count()-1)):   # keep the stretch at end
            w = self.scroll_v.itemAt(i).widget()
            if w: w.setParent(None)
        self._log_line({"type":"system","event":"clear","ts":self._ts()})

    def _persist_feedback(self, thumbs_up: bool):
        os.makedirs('logs', exist_ok=True)
        try:
            if not getattr(self, "_feedback_writer", None):
                self._feedback_writer = AtomicJsonlWriter('logs/feedback.jsonl')
            entry = {'ts': self._ts(), 'conversation': getattr(self, '_conversation_id', None), 'thumbs_up': bool(thumbs_up)}
            self._feedback_writer.write(entry)
            # brief UI acknowledgement
            self.statusBar().showMessage('Feedback saved', 2000)
        except Exception as e:
            self.statusBar().showMessage(f'Feedback log error: {e}', 3000)

    def save_transcript(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save transcript", self._suggest_log_name(".txt"), "Text Files (*.txt)")
        if not path: return
        try:
            atomic_write_text(path, self._read_current_transcript(), encoding="utf-8")
            QMessageBox.information(self, "Saved", f"Transcript saved to:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _read_current_transcript(self) -> str:
        texts = []
        for i in range(self.scroll_v.count()-1):
            w = self.scroll_v.itemAt(i).widget()
            if isinstance(w, Bubble):
                who = "You" if w.me else "AI"
                texts.append(f"{who}: {w.text}")
        return "\n".join(texts)

    def _ensure_logfile(self):
        os.makedirs("logs", exist_ok=True)
        self._log_path = self._suggest_log_name(".jsonl")
        self._log_writer = AtomicJsonlWriter(self._log_path)
        self._log_line({"type":"system","event":"open","ts":self._ts()})

    def _suggest_log_name(self, ext: str) -> str:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join("logs", f"chat_{stamp}{ext}")

    def _ts(self) -> str:
        return datetime.datetime.now().isoformat(timespec="seconds")

    def _log_line(self, obj: Dict[str, Any]):
        try:
            self._log_writer.write(obj)
        except Exception:
            pass

    def add_bubble(self, text: str, me: bool):
        b = Bubble(text, me=me, light=self._light)
        self.scroll_v.insertWidget(self.scroll_v.count()-1, b)
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))

    def send_message(self):
        if self._safe_mode:
            self._disable_input_for_safe_mode(); return
        if getattr(self, "_reply_in_flight", False):
            self.statusBar().showMessage("A reply is already in progress.", 2000)
            return
        text = self.input.toPlainText().strip()
        if not text: return
        self.input.clear()
        self.add_bubble(text, me=True)
        self._log_line({"type":"user","ts":self._ts(),"text":text})
        self._ai_typing(text)

    def _ai_typing(self, prompt: str):
        # typing indicator
        self._typing_lbl = QLabel("Assistant is typing...")
        self._typing_lbl.setStyleSheet("margin: 4px 16px; color:#8aa0b3;")
        self.scroll_v.insertWidget(self.scroll_v.count()-1, self._typing_lbl)
        self._reply_in_flight = True
        self.input.setDisabled(True)
        self.send_btn.setDisabled(True)
        QTimer.singleShot(50, lambda: self._start_reply_worker(prompt))

    # ---------- Guarded reply (non-agentic) ----------
    def _start_reply_worker(self, prompt: str):
        self._reply_thread = QThread(self)
        self._reply_worker = ReplyWorker(self, prompt)
        self._reply_worker.moveToThread(self._reply_thread)
        self._reply_thread.started.connect(self._reply_worker.run)
        self._reply_worker.finished.connect(self._on_reply_ready)
        self._reply_worker.failed.connect(self._on_reply_failed)
        self._reply_worker.finished.connect(self._cleanup_reply_worker)
        self._reply_worker.failed.connect(self._cleanup_reply_worker)
        self._reply_thread.start()

    def _generate_reply_sync(self, prompt: str) -> tuple[str, float]:
        try:
            # simple safety filters (non-executable, bounded length)
            if any(x in prompt.lower() for x in ["rm -rf", "format c:", "shutdown /s", "powershell -c", "http://", "https://", "curl ", "wget ", "import os", "subprocess"]):
                return ("I can't help with system commands, downloads, or external access. This chat is non-agentic and sandboxed.", 0.99)
            else:
                return self._non_agentic_reply(prompt)
        except Exception:
            raise

    def _on_reply_ready(self, prompt: str, reply: str, conf: float):
        try:
            # cap length - increased for comprehensive responses
            reply = reply.strip()
            if len(reply) > 5000:
                reply = reply[:5000] + " ..."

            # Record conversation for training (if learning enabled)
            if self._learning_enabled and hasattr(self, '_trainer'):
                try:
                    self._trainer.log_interaction(
                        user_text=prompt,
                        model_text=reply,
                        meta={"confidence": conf}
                    )
                    # Periodically retrain patterns (every 20 conversations)
                    if hasattr(self._trainer, '_load_all_events'):
                        events = self._trainer._load_all_events()
                        if len(events) % 20 == 0 and len(events) >= 4:
                            self._trainer.train()
                            print(f"Retrained patterns from {len(events)} conversations")
                except Exception as e:
                    print(f"Training error: {e}")

            self._chat_history.append((prompt, reply))
            badge = f"[confidence: {conf:.2f}] "
            self.add_bubble(badge + reply, me=False)
            self._log_line({"type":"assistant","ts":self._ts(),"text":reply,"confidence":conf})
        except Exception as e:
            _log_crash("reply", e)
            self.statusBar().showMessage(f"Reply error: {type(e).__name__}", 5000)
            self.add_bubble("Internal error handled. See logs/chatbox_crash.log for details.", me=False)
            self._log_line(
                {
                    "type": "system",
                    "event": "reply_error",
                    "ts": self._ts(),
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    def _on_reply_failed(self, prompt: str, error_text: str):
        _log_crash("reply_worker", RuntimeError(error_text))
        self.statusBar().showMessage("Reply error", 5000)
        self.add_bubble("Internal error handled. See logs/chatbox_crash.log for details.", me=False)
        self._log_line(
            {
                "type": "system",
                "event": "reply_error",
                "ts": self._ts(),
                "prompt": prompt,
                "error": error_text,
            }
        )

    def _cleanup_reply_worker(self, *_args):
        if getattr(self, "_typing_lbl", None):
            self._typing_lbl.setParent(None)
            self._typing_lbl = None
        self._reply_in_flight = False
        if not self._safe_mode:
            self.input.setDisabled(False)
            self.send_btn.setDisabled(False)
        if self._reply_thread is not None:
            self._reply_thread.quit()
            self._reply_thread.wait(2000)
            self._reply_thread.deleteLater()
            self._reply_thread = None
        if self._reply_worker is not None:
            self._reply_worker.deleteLater()
            self._reply_worker = None

    def _non_agentic_reply(self, prompt: str) -> tuple[str, float]:
        """
        Quantum AI response using Phase 1 & Phase 2 components:
        - Safety checking, conversation memory, RLHF scoring, domain specialization
        """
        # Use our quantum AI system if available
        if self._quantum_ai_enabled:
            try:
                # Process through conversation manager - this handles safety, memory, and response generation
                conv_result = self._conversation_manager.process_turn(prompt)
                
                response = conv_result.get("response", "I understand your question. I am processing this through advanced safety and reasoning systems.")
                confidence = conv_result.get("confidence", 0.8)

                # RLHF scoring if available
                try:
                    if hasattr(self._rlhf_system, 'score_response'):
                        reward_score = self._rlhf_system.score_response(response)
                        confidence = min(confidence, reward_score)
                except:
                    pass

                return response, confidence

            except Exception as e:
                print(f"Quantum AI error: {e}")
                # Fall back to pattern matching

        # Local LLM path (no network).
        if getattr(self, "_local_llm_enabled", False):
            try:
                base_system_prompt = (
                    "You are a helpful, non-agentic assistant. "
                    "Do not provide commands for downloads, system modification, or external access."
                )
                backend = getattr(self, "_local_backend", "rftmw")
                self._ensure_chat_capable_model()
                if backend == "rftmw" and not getattr(self, "_torch_ok", True):
                    raise RuntimeError("torch import healthcheck failed; refusing to import torch in GUI process")
                if getattr(self, "_agent_mode", False):
                    # Agentic mode: allow a restricted set of local tools (repo-only).
                    system_prompt = self._augment_system_prompt_with_topology(base_system_prompt, prompt)
                    text = self._agentic_local_reply(
                        prompt=prompt,
                        backend=backend,
                        system_prompt=system_prompt,
                    )
                    confidence = 0.72
                    return text, confidence

                if backend == "ollama":
                    from src.apps.ollama_client import ollama_chat

                    system_prompt = self._augment_system_prompt_with_topology(base_system_prompt, prompt)
                    text = ollama_chat(
                        user_text=prompt,
                        history=self._chat_history,
                        system_prompt=system_prompt,
                        model=os.getenv("QUANTONIUM_OLLAMA_MODEL"),
                        temperature=0.7,
                        max_tokens=256,
                    )
                    confidence = 0.60
                elif backend == "rftmw":
                    # "No-build" mode: use local RFT weight file through ai_model_wrapper.
                    # Triggered when QUANTONIUM_RFT_FILE_MODE=1 or a quantum JSON path is set.
                    use_file_mode = (
                        os.getenv("QUANTONIUM_RFT_FILE_MODE", "0") == "1"
                        or bool(os.getenv("QUANTONIUM_QUANTUM_WEIGHTS_JSON", "").strip())
                    )
                    if use_file_mode:
                        from src.apps.ai_model_wrapper import format_prompt_auto, generate_response

                        system_prompt = self._augment_system_prompt_with_topology(base_system_prompt, prompt)
                        full_prompt = format_prompt_auto(
                            prompt,
                            history=self._chat_history,
                            system_prompt=system_prompt,
                            max_turns=6,
                        )
                        text = generate_response(full_prompt, max_tokens=160)
                        confidence = 0.66
                    else:
                        engine = self._load_rftmw_engine()
                        full_prompt = self._build_rftmw_prompt(prompt)
                        max_new_tokens = _env_int("QUANTONIUM_RFTMW_MAX_NEW_TOKENS", 120) or 120
                        temperature = _env_float("QUANTONIUM_RFTMW_TEMPERATURE", 0.7)
                        do_sample = os.getenv("QUANTONIUM_RFTMW_DO_SAMPLE", "1") != "0"
                        text = engine.restore_and_generate(
                            full_prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=do_sample,
                        )
                        confidence = 0.68
                else:
                    raise RuntimeError(f"Unsupported local backend: {backend}")
                # Light post-processing and conservative confidence
                text = (text or "").strip()
                if not text:
                    raise RuntimeError("empty generation")
                return text, confidence
            except Exception as e:
                print(f"Local LLM reply failed: {e}")
                return (
                    f"Local backend error for {backend}:{self._current_chat_model_id() or 'unconfigured'} - {e}",
                    0.0,
                )

        disabled_reason = getattr(self, "_local_llm_disabled_reason", "").strip() or "local backend is disabled"
        return (f"Chat backend unavailable: {disabled_reason}", 0.0)

    def _agentic_local_reply(self, *, prompt: str, backend: str, system_prompt: str) -> str:
        """Restricted local agent loop (repo-only tools)."""
        tools_spec = (
            "You can call tools by responding with a single JSON object:\n"
            '{"tool":"search_repo","args":{"query":"...", "max_results": 30}}\n'
            '{"tool":"read_file","args":{"path":"relative/path.py", "max_chars": 8000}}\n'
            '{"tool":"list_files","args":{"dir":"relative/dir", "max_items": 200}}\n'
            "Or respond with normal assistant text when done.\n"
        )

        scratch = ""
        history = self._chat_history[-4:]

        def call_llm(text: str) -> str:
            if backend == "ollama":
                from src.apps.ollama_client import ollama_chat

                return ollama_chat(
                    user_text=text,
                    history=history,
                    system_prompt=system_prompt + "\n\n" + tools_spec,
                    model=os.getenv("QUANTONIUM_OLLAMA_MODEL"),
                    temperature=_env_float("QUANTONIUM_AGENT_TEMPERATURE", 0.4),
                    max_tokens=_env_int("QUANTONIUM_AGENT_MAX_TOKENS", 512) or 512,
                )

            # Fallback to HF local wrapper (less reliable for tool calling).
            from src.apps.ai_model_wrapper import format_prompt_auto, generate_response

            full_prompt = format_prompt_auto(
                text,
                history=history,
                system_prompt=system_prompt + "\n\n" + tools_spec,
                max_turns=6,
            )
            return generate_response(
                full_prompt,
                max_tokens=_env_int("QUANTONIUM_AGENT_MAX_TOKENS", 256) or 256,
                temperature=_env_float("QUANTONIUM_AGENT_TEMPERATURE", 0.7),
            )

        for _ in range(int(self._agent_max_steps)):
            agent_prompt = (
                "User request:\n"
                f"{prompt.strip()}\n\n"
                "Scratchpad:\n"
                f"{scratch}\n"
                "Decide next action. If using a tool, output ONLY the JSON tool call object.\n"
            )
            raw = (call_llm(agent_prompt) or "").strip()
            tool_call = _extract_first_json_object(raw)
            if not tool_call or "tool" not in tool_call:
                # Treat as final answer.
                return raw

            tool = str(tool_call.get("tool", "")).strip()
            args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
            obs = ""
            try:
                if tool == "search_repo":
                    query = str(args.get("query", "")).strip()
                    max_results = int(args.get("max_results", 30))
                    ok, out = search_repo(query, max_results=max_results)
                    obs = out
                elif tool == "read_file":
                    path = str(args.get("path", "")).strip()
                    max_chars = int(args.get("max_chars", 8000))
                    ok, out = read_file(path, max_chars=max_chars)
                    obs = out
                elif tool == "list_files":
                    rel_dir = str(args.get("dir", ".")).strip() or "."
                    max_items = int(args.get("max_items", 200))
                    ok, out = list_files(rel_dir, max_items=max_items)
                    obs = out
                else:
                    obs = f"Tool not allowed: {tool}"
            except Exception as e:
                obs = f"Tool error: {type(e).__name__}: {e}"

            scratch += f"\nTool call: {tool} args={args}\nObservation:\n{obs}\n"

        # Max steps reached: ask LLM to summarize based on scratch.
        summary_prompt = (
            "Summarize the best answer for the user based on this scratchpad:\n"
            f"{scratch}\n"
        )
        return (call_llm(summary_prompt) or "").strip()

    def _current_chat_model_id(self) -> str:
        if getattr(self, "_local_backend", "rftmw") == "ollama":
            return os.getenv("QUANTONIUM_OLLAMA_MODEL", "").strip()
        return self._rftmw_model_id.strip()

    def _ensure_chat_capable_model(self) -> None:
        entry = getattr(self, "_current_model_entry", {}) or {}
        if entry and not bool(entry.get("chat_capable", True)):
            raise RuntimeError(
                f"selected model '{entry.get('model_id', '')}' is embedding-only and cannot answer chat prompts"
            )
    
    def _pattern_fallback_reply(self, prompt: str) -> tuple[str, float]:
        """Legacy fallback retained as a stub; the main chatbox is local-model only."""
        return (
            "Legacy pattern fallback has been removed from the main chatbox. "
            "Use the local model selector and Ollama backend instead.",
            0.0,
        )

    def show_training_stats(self):
        """Show quantum AI system statistics."""
        try:
            if self._quantum_ai_enabled:
                safety_stats = self._safety_system.get_safety_stats()
                conv_stats = self._conversation_manager.get_conversation_stats()
                msg = (
                    "Quantum AI System Statistics\n\n"
                    "Safety System:\n"
                    f"- Violations blocked: {safety_stats.get('violations_blocked', 0)}\n"
                    f"- Total checks: {safety_stats.get('total_checks', 0)}\n"
                    "- Safety status: Active\n\n"
                    "Conversation Memory:\n"
                    f"- Current conversation: {conv_stats.get('conversation_id', 'N/A')}\n"
                    f"- Turn count: {conv_stats.get('turn_count', 0)}\n"
                    f"- Active topics: {', '.join(conv_stats.get('active_topics', []))}\n"
                )
            else:
                backend = getattr(self, "_local_backend", "hf")
                msg = f"Quantum AI System not available; using local backend: {backend}"
                if backend == "rftmw":
                    state_hash = self._rftmw_provenance.get("state_dict_hash_sha256")
                    model_name = self._rftmw_provenance.get("model_name", self._rftmw_model_id)
                    if state_hash:
                        msg += (
                            "\n\nRFTMW Provenance:\n"
                            f"Model: {model_name}\n"
                            f"State-dict SHA-256: {state_hash}"
                        )
            QMessageBox.information(self, "Quantum AI Statistics", msg)
        except Exception as e:
            QMessageBox.warning(self, "Quantum AI Stats", f"Error retrieving stats: {e}")

    # ---------- cleanup ----------
    def closeEvent(self, ev):
        try:
            if getattr(self, "_log_writer", None):
                self._log_writer.close()
            if getattr(self, "_feedback_writer", None):
                self._feedback_writer.close()
        except Exception:
            pass
        super().closeEvent(ev)

# ---------- Entrypoint ----------
def main(argv: Optional[List[str]] = None) -> int:
    _install_crash_hooks()

    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--quit-after",
        type=float,
        default=None,
        help="For smoke tests: close the window after N seconds.",
    )
    args, _unknown = ap.parse_known_args(argv if argv is not None else sys.argv[1:])

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Chatbox")
    app.setFont(QFont("Segoe UI", 10))
    w = Chatbox()
    print("Chatbox created, showing window...")
    w.show()
    w.raise_()  # Bring to front
    w.activateWindow()  # Make it active
    print("Window should be visible now")

    if args.quit_after is not None and args.quit_after >= 0:
        QTimer.singleShot(int(args.quit_after * 1000), app.quit)

    return int(app.exec_())

if __name__ == "__main__":
    raise SystemExit(main())
