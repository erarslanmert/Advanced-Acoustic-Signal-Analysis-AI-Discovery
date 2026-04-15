import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import re
import csv
import time
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram, stft, butter, filtfilt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
import pandas as pd
import sounddevice as sd
import soundfile as sf
import amms_timing
import psutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QTreeWidget, QTreeWidgetItem,
                             QFileDialog, QInputDialog, QMessageBox, QHeaderView, QMenu,
                             QComboBox, QCheckBox, QGroupBox, QDockWidget, QFormLayout, QGridLayout,
                             QSlider, QProgressDialog, QDialog, QDialogButtonBox, QDoubleSpinBox, QSpinBox, QDateTimeEdit,
                             QTextEdit, QListWidget, QListWidgetItem, QProgressBar, QFrame, QScrollArea, QAbstractItemView)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QDateTime
from PyQt5.QtGui import QColor, QBrush, QPixmap, QIcon, QTransform

# v29: AI Bot Integration
try:
    from AIBotManager import AIBotManager, AIBotInferenceThread
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Global reference pool to prevent Python's garbage collector from destroying spawned secondary GUI windows
global_app_instances = []

# ─────────────────────────────────────────────────────────────
#  Utility for PyInstaller Bundling
# ─────────────────────────────────────────────────────────────
def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# ─────────────────────────────────────────────────────────────
#  Channel definitions
#  Order in the 4-ch WAV:  col0=p  col1=x  col2=y  col3=z
# ─────────────────────────────────────────────────────────────
CH_INFO = [
    {"name": "p", "color": "black",  "idx": 0},
    {"name": "x", "color": "blue",   "idx": 1},
    {"name": "y", "color": "red",    "idx": 2},
    {"name": "z", "color": "green",  "idx": 3},
]

SPEC_NFFT     = 4096          # STFT window size (higher = better freq resolution)
SPEC_OVERLAP  = SPEC_NFFT * 3 // 4  # 75% overlap for smoother time axis
MAX_SPEC_COLS = 8000          # boost to 8000 for high detail in long files
ICON_PATH     = get_resource_path("favicon.png")
DT_FMT        = "%Y-%m-%d %H:%M:%S.%f"   # canonical datetime format
QDT_FMT       = "yyyy-MM-dd HH:mm:ss.zzz"  # Qt QDateTimeEdit display format
# v32.14: MAX_WINDOW_DURATION replaced by adaptive self.get_max_win_dur()


class AnnotationEditDialog(QDialog):
    """Custom dialog for creating or editing an annotation.
    Exposes: class_label (str), weight (float 0-10), start_s (float), end_s (float), start_samp (int), end_samp (int).
    Pass start_s / end_s as None to hide the time fields (add-mode).
    """
    def __init__(self, parent=None, class_label="", weight=0.0,
                 start_s=None, end_s=None, start_samp=None, end_samp=None,
                 sample_rate=16000, hardware_offset=0):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.hardware_offset = hardware_offset
        self._block_logic = False # Prevent feedback loops during sync

        self.setWindowTitle("Annotation Properties")
        self.setWindowIcon(QIcon(ICON_PATH))
        self.setMinimumWidth(480)

        layout = QFormLayout(self)

        # Class label
        self.le_class = QLineEdit(class_label)
        layout.addRow("Class Label:", self.le_class)

        # Weight spinner  0..10, step 0.1
        self.sb_weight = QDoubleSpinBox()
        self.sb_weight.setRange(0.0, 10.0)
        self.sb_weight.setSingleStep(0.1)
        self.sb_weight.setDecimals(2)
        self.sb_weight.setValue(float(weight))
        layout.addRow("Weight (0 - 10):", self.sb_weight)

        # Time/Sample fields (only shown when editing an existing annotation)
        self._has_detailed = start_s is not None
        if self._has_detailed:
            separator_1 = QFrame(); separator_1.setFrameShape(QFrame.HLine); separator_1.setFrameShadow(QFrame.Sunken)
            layout.addRow(separator_1)

            # Start Point
            self.sb_start_t = QDoubleSpinBox()
            self.sb_start_t.setRange(0.0, 9999999.0); self.sb_start_t.setDecimals(3); self.sb_start_t.setSuffix(" s")
            self.sb_start_t.setValue(float(start_s))
            layout.addRow("Start Time (rel):", self.sb_start_t)

            self.sb_start_sam = QDoubleSpinBox() # Using Double for large ranges
            self.sb_start_sam.setRange(0, 9e12); self.sb_start_sam.setDecimals(0); self.sb_start_sam.setGroupSeparatorShown(True)
            self.sb_start_sam.setValue(float(start_samp))
            layout.addRow("Start Sample (abs):", self.sb_start_sam)

            separator_2 = QFrame(); separator_2.setFrameShape(QFrame.HLine); separator_2.setFrameShadow(QFrame.Sunken)
            layout.addRow(separator_2)

            # End Point
            self.sb_end_t = QDoubleSpinBox()
            self.sb_end_t.setRange(0.0, 9999999.0); self.sb_end_t.setDecimals(3); self.sb_end_t.setSuffix(" s")
            self.sb_end_t.setValue(float(end_s))
            layout.addRow("End Time (rel):", self.sb_end_t)

            self.sb_end_sam = QDoubleSpinBox()
            self.sb_end_sam.setRange(0, 9e12); self.sb_end_sam.setDecimals(0); self.sb_end_sam.setGroupSeparatorShown(True)
            self.sb_end_sam.setValue(float(end_samp))
            layout.addRow("End Sample (abs):", self.sb_end_sam)

            # Bi-directional sync
            self.sb_start_t.valueChanged.connect(self._sync_start_from_time)
            self.sb_start_sam.valueChanged.connect(self._sync_start_from_samp)
            self.sb_end_t.valueChanged.connect(self._sync_end_from_time)
            self.sb_end_sam.valueChanged.connect(self._sync_end_from_samp)

        # OK / Cancel
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        btns.accepted.connect(self._validate)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def _sync_start_from_time(self, val):
        if self._block_logic: return
        self._block_logic = True
        self.sb_start_sam.setValue(val * self.sample_rate + self.hardware_offset)
        self._block_logic = False

    def _sync_start_from_samp(self, val):
        if self._block_logic: return
        self._block_logic = True
        self.sb_start_t.setValue((val - self.hardware_offset) / self.sample_rate)
        self._block_logic = False

    def _sync_end_from_time(self, val):
        if self._block_logic: return
        self._block_logic = True
        self.sb_end_sam.setValue(val * self.sample_rate + self.hardware_offset)
        self._block_logic = False

    def _sync_end_from_samp(self, val):
        if self._block_logic: return
        self._block_logic = True
        self.sb_end_t.setValue((val - self.hardware_offset) / self.sample_rate)
        self._block_logic = False

    def _validate(self):
        if not self.le_class.text().strip():
            QMessageBox.warning(self, "Input Error", "Class label cannot be empty.")
            return
        if self._has_detailed and self.sb_end_t.value() <= self.sb_start_t.value():
            QMessageBox.warning(self, "Input Error", "End point must be after start point.")
            return
        self.accept()

    @property
    def class_label(self): return self.le_class.text().strip()
    @property
    def weight(self):      return self.sb_weight.value()
    @property
    def start_s(self):     return self.sb_start_t.value() if self._has_detailed else None
    @property
    def end_s(self):       return self.sb_end_t.value() if self._has_detailed else None
    @property
    def start_samp(self):  return int(self.sb_start_sam.value()) if self._has_detailed else None
    @property
    def end_samp(self):    return int(self.sb_end_sam.value()) if self._has_detailed else None


class AudioPCMStreamer(QThread):
    """Secondary thread to lazily load 2GB+ PCM data for playback without blocking the UI."""
    finished = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)
    error    = pyqtSignal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            self.progress.emit(0)
            sr, data = wavfile.read(self.filepath)
            
            # Normalize to float32 (matches the app's internal pipeline)
            # v26.5: Re-restored raw units for AMMS compatibility. 
            data = data.astype(np.float32)
            
            if data.ndim == 1:
                data = data[:, np.newaxis]
                
            self.progress.emit(100)
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class ResourceMonitorThread(QThread):
    """Background thread to monitor RAM and CPU without jittering the main GUI thread."""
    stats_updated = pyqtSignal(float, float, float) # (cpu_pct, ram_avail_gb, ram_total_gb)

    def __init__(self):
        super().__init__()
        self._keep_running = True

    def run(self):
        while self._keep_running:
            try:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                # Emit stats: CPU%, Available GB, Total GB
                self.stats_updated.emit(cpu, mem.available / (1024**3), mem.total / (1024**3))
            except: pass
            self.msleep(2000) # Check every 2 seconds

    def stop(self):
        self._keep_running = False
        self.wait()


# ─────────────────────────────────────────────────────────
#  AI Icons
# ─────────────────────────────────────────────────────────
AI_ICON_PATH = get_resource_path("ai_icon_glow.png")

# ─────────────────────────────────────────────────────────
#  AI Discovery Progress Dialog
# ─────────────────────────────────────────────────────────
class AIProgressDialog(QDialog):
    """v36.5: Premium Interactive progress dashboard matching user reference image."""
    def __init__(self, title, status_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(550)
        self.setMinimumHeight(480)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 1. Header: AI Iconic Animation (Centered)
        icon_row = QHBoxLayout()
        self.lbl_icon = QLabel()
        self.lbl_icon.setFixedSize(64, 64)
        if os.path.exists(AI_ICON_PATH):
             self.orig_pixmap = QPixmap(AI_ICON_PATH).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
             self.lbl_icon.setPixmap(self.orig_pixmap)
        icon_row.addStretch()
        icon_row.addWidget(self.lbl_icon)
        icon_row.addStretch()
        layout.addLayout(icon_row)
        
        # 2. Status Label (Centered, Blue)
        self.lbl_status = QLabel(status_text)
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("font-size: 14pt; font-weight: bold; color: #2196F3;")
        layout.addWidget(self.lbl_status)
        
        # 3. Progress Bar (with text percentage)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid grey; border-radius: 5px; text-align: center; height: 25px; }
            QProgressBar::chunk { background-color: #2196F3; width: 10px; }
        """)
        layout.addWidget(self.progress_bar)
        
        # 4. Console/Log View (Professional Mono)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.WidgetWidth)
        self.log_view.setStyleSheet("""
            background-color: #F9F9F9; 
            border: 1px solid #E0E0E0; 
            font-family: 'Consolas', 'Courier New'; 
            font-size: 9pt; 
            color: #333333;
        """)
        layout.addWidget(self.log_view)
        
        # 5. Background Button
        self.btn_bg = QPushButton("Stay in Background")
        self.btn_bg.setFixedHeight(35)
        self.btn_bg.setStyleSheet("font-weight: bold; background-color: #FFFFFF; border: 1px solid #BDBDBD;")
        self.btn_bg.clicked.connect(self.hide)
        layout.addWidget(self.btn_bg)
        
        # Pulsing Animation Timer
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate_pulse)
        self.pulse_val = 0
        self.pulse_dir = 1
        self.anim_timer.start(50)

    def setValue(self, val):
        self.progress_bar.setValue(val)

    def setLabelText(self, text):
        self.lbl_status.setText(text)
        
        # Animation
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate_pulse)
        self.pulse_val = 0
        self.pulse_dir = 1

    def _animate_pulse(self):
        """Pulsing opacity effect for the AI icon."""
        self.pulse_val += 5 * self.pulse_dir
        if self.pulse_val >= 100 or self.pulse_val <= 0:
             self.pulse_dir *= -1
        
        # We can also do a slight rotation
        self.lbl_icon.setWindowOpacity(0.5 + (self.pulse_val / 200.0))
        # Simple CSS opacity/pulse or transform
        alpha = int(155 + (self.pulse_val))
        # (For simplicity here, just changing label visibility would be jittery, 
        # so let's stick to a robust rotation)
        transform = QTransform().rotate(self.pulse_val / 2.0)
        if hasattr(self, 'orig_pixmap'):
             rotated = self.orig_pixmap.transformed(transform, Qt.SmoothTransformation)
             self.lbl_icon.setPixmap(rotated)

    def showEvent(self, event):
        self.anim_timer.start(50)
        super().showEvent(event)
        
    def hideEvent(self, event):
        self.anim_timer.stop()
        super().hideEvent(event)

# ─────────────────────────────────────────────────────────
#  AI Discovery Results Dialog
# ─────────────────────────────────────────────────────────
class AIDiscoveryDialog(QDialog):
    """Popup that shows discovered time intervals and allows filtering and approval."""
    def __init__(self, raw_intervals, parent=None):
        super().__init__(parent)
        self.raw_intervals = raw_intervals # List of (s, e, "Label|Score") - non-merged or minimally merged
        self.setWindowTitle("AI Discovery - Potential Events")
        self.resize(900, 600)
        
        main_layout = QHBoxLayout(self)
        
        # --- LEFT PANEL: FILTERS ---
        filter_panel = QVBoxLayout()
        filter_panel.addWidget(QLabel("<b>Filters</b>"))
        
        form = QFormLayout()
        
        self.spin_min_dur = QDoubleSpinBox()
        self.spin_min_dur.setRange(0.0, 3600.0)
        self.spin_min_dur.setValue(0.5)
        
        self.spin_max_dur = QDoubleSpinBox()
        self.spin_max_dur.setRange(0.0, 3600.0)
        self.spin_max_dur.setValue(3600.0)
        
        self.spin_min_confidence = QSpinBox()
        self.spin_min_confidence.setRange(0, 100)
        self.spin_min_confidence.setValue(50)
        self.spin_min_confidence.setSuffix("%")
        
        self.spin_merge_gap = QDoubleSpinBox()
        self.spin_merge_gap.setRange(0.0, 10.0)
        self.spin_merge_gap.setValue(1.0)
        self.spin_merge_gap.setSingleStep(0.1)
        self.spin_merge_gap.setSuffix("s")
        
        form.addRow("Merge Gap:", self.spin_merge_gap)
        form.addRow("Min Confidence:", self.spin_min_confidence)
        form.addRow("Min Dur (s):", self.spin_min_dur)
        form.addRow("Max Dur (s):", self.spin_max_dur)
        
        filter_panel.addLayout(form)
        
        filter_panel.addWidget(QLabel("<b>Classes:</b>"))
        self.class_list = QListWidget()
        filter_panel.addWidget(self.class_list)
        
        main_layout.addLayout(filter_panel, 1)
        
        # --- RIGHT PANEL: RESULTS ---
        right_panel = QVBoxLayout()
        
        header_row = QHBoxLayout()
        self.lbl_icon = QLabel()
        if os.path.exists(AI_ICON_PATH):
             self.lbl_icon.setPixmap(QPixmap(AI_ICON_PATH).scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        header_row.addWidget(self.lbl_icon)
        header_row.addWidget(QLabel("<b>Potential Events Discovered:</b>"))
        header_row.addStretch()
        right_panel.addLayout(header_row)
        
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Include", "Event #", "Label", "Time Range", "Confidence"])
        self.tree_widget.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tree_widget.header().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tree_widget.setRootIsDecorated(False)
        
        # INCREASE ROW HEIGHT & PADDING
        self.tree_widget.setStyleSheet("""
            QTreeView::item {
                height: 35px;
                padding-left: 5px;
                border-bottom: 1px solid #EEEEEE;
            }
        """)
        
        self.items_data = [] 
        self.tree_items = []
        
        # Discovery variables
        self.merged_intervals = []
        self.unique_classes = sorted(list(set([x[2].split("|")[0] for x in raw_intervals])))
        
        # Populate class checklist
        for c in self.unique_classes:
            li = QListWidgetItem(c)
            li.setFlags(li.flags() | Qt.ItemIsUserCheckable)
            li.setCheckState(Qt.Checked)
            self.class_list.addItem(li)
            
        right_panel.addWidget(self.tree_widget)
        
        btn_layout = QHBoxLayout()
        self.btn_auto_approve = QPushButton("✅ Auto Approve Selected")
        self.btn_auto_approve.setStyleSheet("font-weight:bold; color:#2E7D32; padding: 8px;")
        btn_layout.addWidget(self.btn_auto_approve)
        
        btn_layout.addStretch()
        
        btns = QDialogButtonBox(QDialogButtonBox.Cancel)
        btns.rejected.connect(self.reject)
        btn_layout.addWidget(btns)
        
        right_panel.addLayout(btn_layout)
        main_layout.addLayout(right_panel, 3)
        
        # Hook up signals
        self.spin_min_dur.valueChanged.connect(self._apply_filters)
        self.spin_max_dur.valueChanged.connect(self._apply_filters)
        self.spin_min_confidence.valueChanged.connect(self._apply_filters)
        self.class_list.itemChanged.connect(self._apply_filters)
        self.spin_merge_gap.valueChanged.connect(self._rebuild_tree)
        
        # Initial Build
        self._rebuild_tree()

    def _rebuild_tree(self):
        """Perform temporal merging logic on raw hits and populate the tree."""
        gap_limit = self.spin_merge_gap.value()
        
        # Raw data is (s, e, "Label|Score")
        # 1. Sort by start time
        sorted_hits = sorted(self.raw_intervals, key=lambda x: x[0])
        
        # 2. Merge by label and gap
        merged = []
        active_trackers = {} # label -> [start, end, max_score]
        
        for s_t, e_t, desc in sorted_hits:
            parts = desc.split("|")
            label = parts[0]
            score = float(parts[1]) if len(parts) > 1 else 0.5
            
            if label in active_trackers:
                track = active_trackers[label]
                if s_t <= track[1] + gap_limit:
                    track[1] = max(track[1], e_t)
                    track[2] = max(track[2], score)
                else:
                    merged.append((track[0], track[1], track[0], track[1], label, track[2]))
                    active_trackers[label] = [s_t, e_t, score]
            else:
                active_trackers[label] = [s_t, e_t, score]
        
        for label, track in active_trackers.items():
            merged.append((track[0], track[1], track[0], track[1], label, track[2]))
            
        # Sort merged result
        self.merged_intervals = sorted(merged, key=lambda x: x[0])
        
        # Update Tree
        self.tree_widget.clear()
        self.tree_items = []
        self.items_data = [] # Store items to return on approval as (s, e, "L|S")
        
        for idx, (s, e, _, _, label, score) in enumerate(self.merged_intervals):
            score_pct = int(score * 100)
            time_range = f"{s:.2f}s to {e:.2f}s"
            
            item = QTreeWidgetItem([
                "", f"Event {idx+1}", label, time_range, f"{score_pct}%"
            ])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked)
            
            # Metadata for fast filtering
            item.setData(0, Qt.UserRole, {
                "label": label,
                "dur": e - s,
                "score": score_pct
            })
            
            self.tree_widget.addTopLevelItem(item)
            self.tree_items.append(item)
            self.items_data.append((s, e, f"{label}|{score}"))
            
        self._apply_filters()

    def _apply_filters(self):
        min_d = self.spin_min_dur.value()
        max_d = self.spin_max_dur.value()
        min_c = self.spin_min_confidence.value()
        
        allowed_classes = set()
        for i in range(self.class_list.count()):
            li = self.class_list.item(i)
            if li.checkState() == Qt.Checked:
                allowed_classes.add(li.text())
        
        for item in self.tree_items:
            meta = item.data(0, Qt.UserRole)
            
            vis = True
            if meta["label"] not in allowed_classes: vis = False
            elif meta["dur"] < min_d or meta["dur"] > max_d: vis = False
            elif meta["score"] < min_c: vis = False
            
            item.setHidden(not vis)
            
    def get_selected_intervals(self):
        selected = []
        for i, item in enumerate(self.tree_items):
            if not item.isHidden() and item.checkState(0) == Qt.Checked:
                 selected.append(self.items_data[i])
        return selected

class SpectralRefineThread(QThread):
    """Refinement thread to produce high-res HD spectral data for a specific time segment."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, filepath, start_time, end_time, sr, n_channels, audio_data=None, target_cols=6000, is_auto=False):
        super().__init__()
        self.filepath = filepath
        self.start_time = max(0.0, start_time)
        self.end_time = end_time
        self.sr = sr
        self.n_channels = n_channels
        self.audio_data = audio_data # v24: Direct RAM slice if available
        self.target_cols = target_cols # v27.8: Dynamic resolution
        self.is_auto = is_auto

    def run(self):
        try:
            if self.audio_data is not None:
                # v24: Fast path - data is already in RAM!
                data = self.audio_data
            else:
                # v23: Switch to soundfile for fast surgical extraction
                st_f = int(self.start_time * self.sr)
                en_f = int(self.end_time * self.sr)
                fr   = max(0, en_f - st_f)
                data, _ = sf.read(self.filepath, start=st_f, frames=fr, dtype='float32', always_2d=True)
                
            n_ch = data.shape[1]
            dur = data.shape[0] / self.sr
            
            # Perform high-resolution HD STFT for this small slice
            # v27.8: Use the dynamic resolution from the UI
            target_cols = self.target_cols
            
            if n_ch >= 3:
                rgb, intensity, f_bins, t_spec = AudioAnnotatorApp._complex_intensity_rgb(
                    data, self.sr, force_max_cols=target_cols)
                payload = { 
                    "rgb": rgb, "intensity": intensity, 
                    "f": f_bins, "t": t_spec + self.start_time, "is_mono": False 
                }
            else:
                f_bins, t_spec, sxx = AudioAnnotatorApp._draw_power_spectrogram_core(
                    data[:, 0], self.sr, dur)
                payload = { "mono": sxx, "f": f_bins, "t": t_spec + self.start_time, "is_mono": True }
            
            self.finished.emit(payload)
        except Exception as e:
            self.error.emit(str(e))


class AudioLoaderThread(QThread):
    progress = pyqtSignal(int, str)
    finished_success = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            # 0% - 10% : Initial Disk I/O & Cache Discovery
            self.progress.emit(2, "Checking for spectral cache...")
            cache_file = self.filepath.lower().replace(".wav", ".spec.npz")
            mtime = os.path.getmtime(self.filepath)
            size  = os.path.getsize(self.filepath)
            
            payload = None
            
            # --- Try loading from Persistent Cache ---
            if os.path.exists(cache_file):
                try:
                    self.progress.emit(5, "Validating and loading cached spectral data...")
                    with np.load(cache_file, allow_pickle=True) as arc:
                        # Integrity check: mtime, size, and NFFT must match
                        if (arc["mtime"] == mtime and arc["size"] == size and 
                            arc.get("nfft", SPEC_NFFT) == SPEC_NFFT and
                            "wave_envelope" in arc):
                            
                            self.progress.emit(10, "Cache-hit: Unpacking high-res matrices...")
                            rgb_base = arc["rgb_base"]
                            intensity_base = arc["intensity_base"] if "intensity_base" in arc else None
                            f_bins   = arc["f_bins"]
                            t_spec   = arc["t_spec"]
                            mono_base = arc["mono_base"] if "mono_base" in arc else None
                            mono_f_bins = arc["mono_f_bins"] if "mono_f_bins" in arc else None
                            
                            # v9 Upgrade: Load the pre-calculated waveform envelope
                            wave_envelope = arc["wave_envelope"] if "wave_envelope" in arc else None
                            sr_cached = int(arc["sr"]) if "sr" in arc else 16000
                            dur_cached = float(arc["duration"]) if "duration" in arc else 0.0
                            nch_cached = int(arc["n_channels"]) if "n_channels" in arc else 0
                            
                            # DECOUPLED LOADING: We return the payload WITHOUT the 2GB 'data'
                            # The main app will spawn a background streamer for 'data'
                            # v14 Upgrade: Detect if this is an 'Overview' or 'Master' HD cache
                            # If t_spec has significantly more columns than MAX_SPEC_COLS, it's Master HD.
                            is_master = len(t_spec) > (MAX_SPEC_COLS * 1.5)
                            
                            payload = {
                                "filepath": self.filepath, "sr": sr_cached, "data": None,
                                "wave_envelope": wave_envelope,
                                "n_channels": nch_cached, "duration": dur_cached,
                                "rgb_base": rgb_base, "intensity_base": intensity_base, 
                                "f_bins": f_bins, "t_spec": t_spec,
                                "mono_base": mono_base, "mono_f_bins": mono_f_bins,
                                "from_cache": True, "is_master_hd": is_master
                            }
                            self.progress.emit(100, "Metadata Ready! Activating UI while audio buffers...")
                except Exception as ce:
                    self.progress.emit(5, f"Cache stale or corrupt ({ce}), falling back to recalculation...")
                    payload = None

            # --- Full Spectral Recalculation (Cache-miss) ---
            if payload is None:
                self.progress.emit(10, f"Reading {os.path.basename(self.filepath)} from disk...")
                sr, data = wavfile.read(self.filepath)
                
                # 10% - 20% : PCM Preparation
                self.progress.emit(12, "Checking audio bit-depth...")
                # v26.5: Re-restored raw units for AMMS compatibility (prevents saturation)
                data = data.astype(np.float32)

                if data.ndim == 1:
                    data = data[:, np.newaxis]

                n_channels = data.shape[1]
                duration = data.shape[0] / sr
                
                # v9 Upgrade: Pre-calculate the Waveform Envelope (100k points)
                self.progress.emit(18, "Generating waveform envelope for instant recall...")
                env_size = 100000
                wave_envelope = np.zeros((min(env_size, data.shape[0]), n_channels), dtype=np.float32)
                step = max(1, data.shape[0] // env_size)
                for c in range(n_channels):
                    wave_envelope[:, c] = data[::step, c][:wave_envelope.shape[0]]

                # 20% - 95% : Multi-Channel Spectral Heavy Lifting
                def prog_proxy(val, txt):
                    mapped = int(20 + (val * 75 / 100))
                    self.progress.emit(mapped, txt)

                rgb_base, f_bins, t_spec = None, None, None
                mono_base, mono_f_bins = None, None

                if n_channels >= 3:
                    try:
                        rgb, intensity, fb, ts = AudioAnnotatorApp._complex_intensity_rgb(data, sr, progress_cb=prog_proxy)
                        rgb_base = rgb.copy()
                        intensity_base = intensity.copy()
                        f_bins = fb
                        t_spec = ts
                    except Exception as e:
                        self.progress.emit(25, f"HSV match failure ({e}), falling back to mono...")
                        fb, ts, Sxx = AudioAnnotatorApp._draw_power_spectrogram_core(data[:, 0], sr, duration, progress_cb=prog_proxy)
                        mono_base = Sxx
                        mono_f_bins = fb
                else:
                    self.progress.emit(20, "Single-channel detected, calculating power spectrogram...")
                    fb, ts, Sxx = AudioAnnotatorApp._draw_power_spectrogram_core(data[:, 0], sr, duration, progress_cb=prog_proxy)
                    mono_base = Sxx
                    mono_f_bins = fb

                # Finalize
                payload = {
                    "filepath": self.filepath, "sr": sr, "data": data,
                    "wave_envelope": wave_envelope,
                    "n_channels": n_channels, "duration": duration,
                    "rgb_base": rgb_base, "intensity_base": intensity_base, 
                    "f_bins": f_bins, "t_spec": t_spec,
                    "mono_base": mono_base, "mono_f_bins": mono_f_bins,
                    "from_cache": False, "is_master_hd": False
                }
                
                # --- Silently Save to Cache (v2 format) ---
                try:
                    self.progress.emit(98, "Silently caching spectral results for future visits...")
                    np.savez_compressed(
                        cache_file,
                        mtime=mtime, size=size, nfft=SPEC_NFFT,
                        sr=sr, duration=duration, n_channels=n_channels,
                        wave_envelope=wave_envelope,
                        rgb_base=rgb_base, intensity_base=intensity_base,
                        f_bins=f_bins, t_spec=t_spec,
                        mono_base=mono_base, mono_f_bins=mono_f_bins
                    )
                except Exception as se:
                    print(f"Failed to save spectral cache: {se}")

                self.progress.emit(100, "Audio environment ready!")

            self.finished_success.emit(payload)
            
        except Exception as e:
            self.error.emit(str(e))


class FullFileHDRefiner(QThread):
    """v14: Ground-truth HD refiner that calculates the entire file at 8 cols/sec."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str) # file path

    def __init__(self, filepath, sr, n_channels):
        super().__init__()
        self.filepath = filepath
        self.sr = sr
        self.n_channels = n_channels

    def run(self):
        # v32.0: Adaptive Resolution Fallback
        # If 8-cols/sec fails (due to 313 MiB allocation errors), we retry at 4-cols/sec.
        fallback_attempt = False
        target_hz = 8
        
        while True:
            try:
                self.progress.emit(5, "Reading audio structure...")
                _, data_mmap = wavfile.read(self.filepath, mmap=True)
                
                def prog_proxy(val, txt):
                    self.progress.emit(val, txt)

                target_hd_cols = int((data_mmap.shape[0] / self.sr) * target_hz)
                self.progress.emit(10, f"Calculating Full-File HD Matrix ({target_hd_cols} columns)...")
                
                rgb, intensity, fb, ts = AudioAnnotatorApp._complex_intensity_rgb(
                    data_mmap, self.sr, progress_cb=prog_proxy, force_max_cols=target_hd_cols
                )
                
                self.progress.emit(95, "Updating Master HD Cache on disk...")
                cache_file = self.filepath.lower().replace(".wav", ".spec.npz")
                mtime = os.path.getmtime(self.filepath)
                size  = os.path.getsize(self.filepath)
                
                # High-res envelope
                env_size = 200000 
                wave_envelope = np.zeros((min(env_size, data_mmap.shape[0]), self.n_channels), dtype=np.float32)
                step = max(1, data_mmap.shape[0] // env_size)
                for c in range(self.n_channels):
                    wave_envelope[:, c] = data_mmap[::step, c][:wave_envelope.shape[0]].astype(np.float32)

                np.savez_compressed(
                    cache_file,
                    mtime=mtime, size=size, nfft=SPEC_NFFT,
                    sr=self.sr, duration=(data_mmap.shape[0]/self.sr), n_channels=self.n_channels,
                    wave_envelope=wave_envelope,
                    rgb_base=rgb, intensity_base=intensity,
                    f_bins=fb, t_spec=ts,
                    mono_base=None, mono_f_bins=None
                )
                
                self.progress.emit(100, f"Full-File HD Upgrade Complete ({target_hz} cols/sec).")
                self.finished.emit(self.filepath)
                break # Success!

            except (MemoryError, np.core._exceptions._ArrayMemoryError) as me:
                if not fallback_attempt:
                    print(f"Master HD Memory Limit Reached at {target_hz} cols/sec. Retrying at half resolution...")
                    self.progress.emit(10, "Memory limit reached. Retrying at 50% resolution...")
                    target_hz = 4
                    fallback_attempt = True
                    # Clean up what we can before retry
                    try: del rgb, intensity, fb, ts, wave_envelope
                    except: pass
                    continue
                else:
                    print(f"Master HD Fatal Memory Error: {me}")
                    self.progress.emit(0, "Master HD Upgrade Failed: Out of Memory.")
                    break
            except Exception as e:
                print(f"Master HD Upgrade Error: {e}")
                break


class AudioAnnotatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Acoustic Event Pipeline Annotator Pro")
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowIcon(QIcon(ICON_PATH))

        # ── audio state ──────────────────────────────────────
        self.audio_data   = None      # np.ndarray [nSamples x nCh]  (float64)
        self.sample_rate  = None
        self.n_channels   = 0
        
        # v27.9: Track preferred HD resolution globally so it stays 'sticky' between windows
        if not hasattr(AudioAnnotatorApp, 'GLOBAL_HD_COLS'):
            AudioAnnotatorApp.GLOBAL_HD_COLS = 6000

        # ── annotation state ─────────────────────────────────
        self.annotations  = []
        self.spans        = []        # list of (waveform_span, text, spec_span)
        self.ghost_spans  = []        # temporary AI discovery highlights
        self._cached_noise_sample = None  # Cache for locked noise profiles

        # ── playback state ───────────────────────────────────
        self.play_marker         = None
        self.play_text           = None
        self.spec_marker         = None   # vertical line on spectrogram
        self.bg                  = None
        self.playback_timer      = QTimer()
        self.playback_timer.timeout.connect(self.update_playhead)
        self.playback_start_x    = 0.0
        self.playback_start_sys  = 0.0
        self.playback_speed      = 1.0
        self.paused              = False
        self.paused_x            = 0.0
        # v31.0: Latency Compensation (seconds)
        # Accounts for Windows MME / Bluetooth buffering delays (MOMENTUM 4).
        self.latency_compensation = 0.65 

        # Debounce timer: refreshes the blit cache 200 ms after the last scroll
        self._bg_debounce        = QTimer()
        self._bg_debounce.setSingleShot(True)
        self._bg_debounce.timeout.connect(self.update_background)
        
        # v10 Upgrade: Debounce timer for HD Spectral Refinement
        self.refine_timer = QTimer()
        self.refine_timer.setSingleShot(True)
        self.refine_timer.timeout.connect(lambda: self.trigger_refinement(is_auto=True))
        self.current_refine_thread = None
        self.is_view_refined = False
        self.last_refined_xlim = (None, None)
        self.is_pcm_buffering = False
        
        # v12 Upgrade: Predictive RAM Cache
        # Stores (start, end) tuples mapping to STFT payloads for instant recall
        self.refined_cache = {} 
        self.predictive_timer = QTimer()
        self.predictive_timer.setSingleShot(True)
        self.predictive_timer.timeout.connect(self.start_predictive_caching)
        self.predictive_thread = None
        self.master_hd_refiner = None
        self.is_master_hd      = False
        self.is_upgrading_hd   = False
        self.master_hd_data    = None 
        self._spec_active_rgb  = None # v16.1: Tracks exactly what is currently in imshow
        self._spec_active_mono = None

        # ── color palette ────────────────────────────────────
        self.class_colors  = {}
        color_list = (list(mcolors.TABLEAU_COLORS.values()) +
                      ['purple', 'teal', 'magenta', 'olive', 'navy', 'brown', 'maroon', 'indigo'])
        self.available_colors = color_list
        self.color_index      = 0

        # --- Animation State for Highlights ---
        self.active_fades = {} # dict of idx -> current_alpha
        self.fade_timer = QTimer()
        self.fade_timer.timeout.connect(self.process_fades)

        # ── AMMS Timing state ────────────────────────────────
        self.amms_timing = None
        self.listen_ref_x = 0.0  # reference points for relative top axis

        # v29: AI Bot State
        self.ai_progress_dialog = None
        
        # v32.14: Resource Monitoring Thread
        self.latest_resources = (0.0, 16.0, 16.0) # Defaults
        self.res_monitor = ResourceMonitorThread()
        self.res_monitor.stats_updated.connect(self._on_resource_stats)
        self.res_monitor.start()

        self.setup_ui()

    # ─────────────────────────────────────────────────────────
    #  UI setup
    # ─────────────────────────────────────────────────────────
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ── Control row ──────────────────────────────────────
        ctrl = QHBoxLayout()
        root.addLayout(ctrl)

        self.btn_load = QPushButton("Load WAV File")
        self.btn_load.clicked.connect(self.load_wav)
        ctrl.addWidget(self.btn_load)

        self.btn_stop = QPushButton("Stop Audio")
        self.btn_stop.clicked.connect(self.stop_audio)
        ctrl.addWidget(self.btn_stop)

        ctrl.addWidget(QLabel("Speed:"))
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.5x", "1.0x", "1.5x", "2.0x", "3.0x", "5.0x", "10.0x"])
        self.combo_speed.setCurrentText("1.0x")
        self.combo_speed.currentTextChanged.connect(self.change_speed)
        ctrl.addWidget(self.combo_speed)

        ctrl.addWidget(QLabel("Recording Start Datetime Override (YYYY-MM-DD HH:MM:SS):"))
        self.datetime_entry = QLineEdit("2015-03-04 18:46:13.000")
        self.datetime_entry.editingFinished.connect(self._on_base_datetime_changed)
        ctrl.addWidget(self.datetime_entry)

        ctrl.addStretch()

        self.btn_export = QPushButton("Export Tracking Spreadsheet")
        self.btn_export.clicked.connect(self.export_data)
        ctrl.addWidget(self.btn_export)

        # ── Channel checkbox row ─────────────────────────────
        ch_row = QHBoxLayout()
        root.addLayout(ch_row)

        ch_group = QGroupBox("Waveform Channels")
        ch_inner = QHBoxLayout(ch_group)
        self.ch_checkboxes = {}
        for ch in CH_INFO:
            cb = QCheckBox(ch["name"])
            cb.setChecked(True)
            style = f"color: {ch['color']}; font-weight: bold;"
            cb.setStyleSheet(style)
            cb.stateChanged.connect(self.update_channel_visibility)
            self.ch_checkboxes[ch["name"]] = cb
            ch_inner.addWidget(cb)
        
        # Add Sync Offset control directly next to channels (v31.2)
        ch_inner.addSpacing(20)
        v_line = QFrame()
        v_line.setFrameShape(QFrame.VLine)
        v_line.setFrameShadow(QFrame.Sunken)
        ch_inner.addWidget(v_line)
        ch_inner.addSpacing(10)
        
        sync_label = QLabel("Sync Offset:")
        sync_label.setStyleSheet("font-weight: bold;")
        ch_inner.addWidget(sync_label)
        
        self.spin_latency = QDoubleSpinBox()
        self.spin_latency.setRange(0.00, 2.00)
        self.spin_latency.setSingleStep(0.05)
        self.spin_latency.setValue(0.65) # User requested 0.65 in their manual diff
        self.spin_latency.setSuffix(" s")
        self.spin_latency.setFixedWidth(80)
        self.spin_latency.setToolTip("Fine-tune the A/V delay. Increase if playhead is ahead of sound (Bluetooth).")
        self.spin_latency.valueChanged.connect(self.update_latency_from_ui)
        ch_inner.addWidget(self.spin_latency)
        
        ch_inner.addSpacing(20)
        v_line2 = QFrame()
        v_line2.setFrameShape(QFrame.VLine)
        v_line2.setFrameShadow(QFrame.Sunken)
        ch_inner.addWidget(v_line2)
        ch_inner.addSpacing(10)
        
        self.check_denoise = QCheckBox("🎙️ Noise Reduction")
        self.check_denoise.setToolTip(
            "Apply spectral noise reduction during playback.\n"
            "Learns background noise from the first 0.5s of the window and subtracts it."
        )
        self.check_denoise.setStyleSheet("font-weight: bold; color: #1565C0;")
        ch_inner.addWidget(self.check_denoise)

        ch_inner.addSpacing(8)
        ch_inner.addWidget(QLabel("Strength:"))
        self.slider_denoise_intensity = QSlider(Qt.Horizontal)
        self.slider_denoise_intensity.setRange(1, 10)
        self.slider_denoise_intensity.setValue(5)
        self.slider_denoise_intensity.setFixedWidth(100)
        self.slider_denoise_intensity.setToolTip(
            "1 = subtle noise reduction\n10 = aggressive suppression"
        )
        ch_inner.addWidget(self.slider_denoise_intensity)

        self.check_lock_noise = QCheckBox("🔒 Fix Profile")
        self.check_lock_noise.setToolTip(
            "Lock the current background noise profile.\n"
            "Unchecked: Update noise fingerprint on every click.\n"
            "Checked: Reuse the last good noise reference found."
        )
        self.check_lock_noise.setStyleSheet("font-size: 8pt; color: #757575;")
        ch_inner.addWidget(self.check_lock_noise)

        ch_inner.addSpacing(15)
        v_line3 = QFrame()
        v_line3.setFrameShape(QFrame.VLine)
        v_line3.setFrameShadow(QFrame.Sunken)
        ch_inner.addWidget(v_line3)
        ch_inner.addSpacing(10)

        self.check_boost = QCheckBox("🚀 Boost (DRC)")
        self.check_boost.setToolTip("Enable Dynamic Range Compression (DRC) to bring up quiet sounds and limit loud ones.\n"
                                    "This makes faint background events much more audible.")
        self.check_boost.setStyleSheet("font-weight: bold; color: #E64A19;")
        ch_inner.addWidget(self.check_boost)

        ch_inner.addSpacing(8)
        ch_inner.addWidget(QLabel("Boost:"))
        self.slider_boost_intensity = QSlider(Qt.Horizontal)
        self.slider_boost_intensity.setRange(1, 10)
        self.slider_boost_intensity.setValue(5)
        self.slider_boost_intensity.setFixedWidth(100)
        self.slider_boost_intensity.setToolTip("1 = subtle cleanup boost\n10 = aggressive compression (brings up every whisper)")
        ch_inner.addWidget(self.slider_boost_intensity)


        # No spectrogram overlay needed for spectral subtraction
        self.denoise_cutoff_line  = None
        self.denoise_highcut_line = None
        self._denoise_band_fill   = None
        self._denoise_label       = None
        self._denoise_label_high  = None


        ch_row.addWidget(ch_group)
        ch_row.addStretch()

        # ── Matplotlib figure: top=waveform, bottom=spectrogram ──
        self.fig = plt.figure(figsize=(12, 7))
        gs = gridspec.GridSpec(2, 1, figure=self.fig,
                               height_ratios=[1, 1.5], hspace=0.3)

        self.ax      = self.fig.add_subplot(gs[0])          # waveform
        self.ax_spec = self.fig.add_subplot(gs[1],
                                            sharex=self.ax)  # spectrogram

        self.ax.set_title("Acoustic Waveform  (Ctrl+Click to play · drag to annotate · Ctrl+Scroll to zoom)", pad=40)
        self.ax.set_ylabel("Amplitude")
        plt.setp(self.ax.get_xticklabels(), visible=False)   # hide x labels on top

        self.ax_spec.set_xlabel("Time (seconds)")
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.ax_spec.set_title("Spectrogram (pressure channel p)", pad=15)

        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, stretch=3)

        # SpanSelector lives on BOTH the waveform and spectrogram axes
        self.span_selector = SpanSelector(
            self.ax, self.on_select_interval, 'horizontal', useblit=True,
            props=dict(alpha=0.3, facecolor='green'), interactive=False
        )
        self.span_selector_spec = SpanSelector(
            self.ax_spec, self.on_select_interval, 'horizontal', useblit=True,
            props=dict(alpha=0.3, facecolor='green'), interactive=False
        )

        # v31.2: Secondary Top Axis for Relative Listening Time
        self.ax_top = self.ax.secondary_xaxis('top', functions=(
            lambda x: x - self.listen_ref_x,
            lambda x: x + self.listen_ref_x
        ))
        self.ax_top.set_xlabel("Relative Time (s) [from listen click]")
        # Color it slightly differently for distinction
        self.ax_top.xaxis.label.set_color('#1976D2')
        self.ax_top.tick_params(axis='x', colors='#1976D2')

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('resize_event', self.on_resize)

        # ── Annotation table ─────────────────────────────────
        self.tree = QTreeWidget()
        self.tree.setColumnCount(8)
        self.tree.setHeaderLabels([
            "Row", "Start Sample", "End Sample", 
            "Start Datetime", "End Datetime", 
            "Class Label", "Color", "Weight"
        ])
        
        # Adjust Header Sizing (Aesthetic Balance v32.10: Centered + Balanced)
        header = self.tree.header()
        header.setStretchLastSection(False) 
        header.setDefaultAlignment(Qt.AlignCenter) # Center all Header Titles
        
        # 1. Pinned / Fixed Columns (Indices 0, 6, 7)
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.resizeSection(0, 50)   # Event #
        header.setSectionResizeMode(6, QHeaderView.Fixed)
        header.resizeSection(6, 60)   # Color (Slightly increased)
        header.setSectionResizeMode(7, QHeaderView.Fixed)
        header.resizeSection(7, 85)   # Weight (Slightly increased)
        
        # 2. Balanced / Stretched Columns (Indices 1, 2, 3, 4, 5)
        # We set these to Stretch so they divide the remaining room equally
        for i in [1, 2, 3, 4, 5]:
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        # v36.1: Allow multi-row selection (Shift/Ctrl)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)
        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)
        root.addWidget(self.tree, stretch=1)

        lbl = QLabel("Right-click to Edit/Delete · Delete key removes selected · "
                     "Ctrl+Click waveform to play · Ctrl+RightClick to pause")
        root.addWidget(lbl)

        # ── Setup Dockable Configuration Window ────────────────
        self.dock = QDockWidget("Wave Time Window Settings", self)
        self.dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        
        # v33.0: 1080p Resolution Fix - Wrap sidebar in a ScrollArea
        # This prevents widgets from being cut off on smaller screens.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        dock_content = QWidget()
        dock_layout = QVBoxLayout(dock_content)
        dock_layout.setContentsMargins(5, 5, 5, 5)
        dock_layout.setSpacing(8)
        
        scroll.setWidget(dock_content)
        self.dock.setWidget(scroll)
        
        # A. Time Axis Controls
        gb_time = QGroupBox("Time Axis (X) Configuration")
        form_time = QFormLayout(gb_time)
        self.entry_start_time = QLineEdit("0.0")
        self.entry_end_time   = QLineEdit("300.0")
        self.entry_duration   = QLineEdit("300.0")
        
        # v32.12: Linked synchronization for Start/End/Duration
        self.entry_start_time.editingFinished.connect(self._on_window_start_changed)
        self.entry_end_time.editingFinished.connect(self._on_window_end_changed)
        self.entry_duration.editingFinished.connect(self._on_window_dur_changed)
        
        form_time.addRow("Window Start (s):", self.entry_start_time)
        form_time.addRow("Window End (s):",   self.entry_end_time)
        form_time.addRow("Window Duration (s):", self.entry_duration)
        
        self.btn_apply_time = QPushButton("Apply Time View")
        self.btn_apply_time.clicked.connect(self.apply_time_window)
        form_time.addRow(self.btn_apply_time)
        
        nav_layout = QHBoxLayout()
        self.btn_prev_win = QPushButton("< Prev Window")
        self.btn_next_win = QPushButton("Next Window >")
        self.btn_prev_win.clicked.connect(self.shift_window_prev)
        self.btn_next_win.clicked.connect(self.shift_window_next)
        nav_layout.addWidget(self.btn_prev_win)
        nav_layout.addWidget(self.btn_next_win)
        form_time.addRow(nav_layout)
        dock_layout.addWidget(gb_time)
        
        # B. Amplitude Controls
        gb_amp = QGroupBox("Waveform Amplitude Limits (Y1)")
        form_amp = QFormLayout(gb_amp)
        self.entry_amp_min = QLineEdit()
        self.entry_amp_min.setText("-3000.0")
        self.entry_amp_max = QLineEdit()
        self.entry_amp_max.setText("3000.0")
        form_amp.addRow("Min Amplitude:", self.entry_amp_min)
        form_amp.addRow("Max Amplitude:", self.entry_amp_max)
        # v26: Added Layout for Apply & Auto buttons
        amp_btn_layout = QHBoxLayout()
        self.btn_apply_amp = QPushButton("Apply Limits")
        self.btn_auto_amp  = QPushButton("Auto-Scale")
        self.btn_apply_amp.clicked.connect(self.apply_amp_limits)
        self.btn_auto_amp.clicked.connect(self.auto_scale_amplitude)
        amp_btn_layout.addWidget(self.btn_apply_amp)
        amp_btn_layout.addWidget(self.btn_auto_amp)
        form_amp.addRow(amp_btn_layout)
        dock_layout.addWidget(gb_amp)
        
        # C. Frequency Controls
        gb_freq = QGroupBox("Spectrogram Frequency Limits (Y2)")
        form_freq = QFormLayout(gb_freq)
        self.entry_freq_min = QLineEdit("0.0")
        self.entry_freq_max = QLineEdit()
        form_freq.addRow("Min Frequency (Hz):", self.entry_freq_min)
        form_freq.addRow("Max Frequency (Hz):", self.entry_freq_max)
        self.btn_apply_freq = QPushButton("Apply Frequency Limits")
        self.btn_apply_freq.clicked.connect(self.apply_freq_limits)
        form_freq.addRow(self.btn_apply_freq)
        dock_layout.addWidget(gb_freq)
        
        # D. Spectrogram Visual Polish Settings
        gb_polish = QGroupBox("Spectrogram Appearance")
        form_polish = QFormLayout(gb_polish)
        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(10, 400) # 0.1x to 4.0x Contrast Multiply
        self.slider_contrast.setValue(100) # 1.0x native center
        self.slider_contrast.setTracking(False) # v10: Refresh only when slider is released for smoothness
        self.slider_contrast.valueChanged.connect(self.update_spec_appearance)
        
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-100, 100) # -1.0 to 1.0 (RGB additive shift bounds)
        self.slider_brightness.setValue(0)
        self.slider_brightness.setTracking(False) # v10: Refresh only when slider is released for smoothness
        self.slider_brightness.valueChanged.connect(self.update_spec_appearance)
        
        form_polish.addRow("Contrast:", self.slider_contrast)
        form_polish.addRow("Brightness:", self.slider_brightness)
        
        # v28.0: Advanced Spectral Enhancements
        self.slider_gamma = QSlider(Qt.Horizontal)
        self.slider_gamma.setRange(10, 300) # 0.1x to 3.0x Gamma (100 = 1.0 base)
        self.slider_gamma.setValue(100)
        self.slider_gamma.setTracking(False)
        self.slider_gamma.valueChanged.connect(self.update_spec_appearance)
        form_polish.addRow("Gamma (Mid-Tones):", self.slider_gamma)
        
        self.slider_saturation = QSlider(Qt.Horizontal)
        self.slider_saturation.setRange(0, 400) # 0.0x to 4.0x Saturation (100 = 1.0 base)
        self.slider_saturation.setValue(100)
        self.slider_saturation.setTracking(False)
        self.slider_saturation.valueChanged.connect(self.update_spec_appearance)
        form_polish.addRow("Saturation Boost:", self.slider_saturation)
        
        # Color Filters & Colormaps
        filter_layout = QHBoxLayout()
        self.btn_reset_spec = QPushButton("↺ Reset All")
        self.btn_reset_spec.setToolTip("Reset all spectrogram visual settings to defaults.")
        self.btn_reset_spec.clicked.connect(self.reset_spec_appearance)
        filter_layout.addWidget(self.btn_reset_spec)
        
        self.chk_invert = QCheckBox("Negative")
        self.chk_invert.stateChanged.connect(self.update_spec_appearance)
        filter_layout.addWidget(self.chk_invert)
        
        self.combo_cmap = QComboBox()
        # v28.1: Added 'Native Bearing' as the first priority option
        self.combo_cmap.addItems(["Native Bearing", "magma", "inferno", "plasma", "viridis", "gray", "hot", "cool"])
        self.combo_cmap.setCurrentText("Native Bearing")
        self.combo_cmap.currentTextChanged.connect(self.update_spec_appearance)
        filter_layout.addWidget(QLabel("Map:"))
        filter_layout.addWidget(self.combo_cmap)
        
        form_polish.addRow(filter_layout)
        
        dock_layout.addWidget(gb_polish)
        
        # v17: Manual Refinement Control
        gb_view = QGroupBox("Viewing Controls")
        layout_view = QVBoxLayout(gb_view)
        
        self.chk_auto_refine = QCheckBox("Auto-Refine Current View")
        self.chk_auto_refine.setChecked(False) # Checked by default
        self.chk_auto_refine.setToolTip("Automatically calculate HD spectrogram for each window you visit.")
        layout_view.addWidget(self.chk_auto_refine)
        
        self.btn_refine_now = QPushButton("🚀 Refine Window Now (HD)")
        self.btn_refine_now.setStyleSheet("font-weight: bold; padding: 5px; color: #2196F3;")
        self.btn_refine_now.clicked.connect(self.manual_refine_now)
        layout_view.addWidget(self.btn_refine_now)
        
        # v27.9: Re-restored with Persistence
        hd_res_layout = QHBoxLayout()
        hd_res_layout.addWidget(QLabel("HD Columns:"))
        self.spin_hd_cols = QSpinBox()
        self.spin_hd_cols.setRange(100, 50000)
        self.spin_hd_cols.setValue(AudioAnnotatorApp.GLOBAL_HD_COLS)
        self.spin_hd_cols.setSingleStep(100)
        self.spin_hd_cols.setToolTip("Set the number of columns for spectral refinement. Higher = clearer but slower.")
        self.spin_hd_cols.valueChanged.connect(self._on_hd_cols_changed)
        hd_res_layout.addWidget(self.spin_hd_cols)
        layout_view.addLayout(hd_res_layout)
        
        dock_layout.addWidget(gb_view)
        
        # v15: Stacked Process Status Indicators
        gb_status = QGroupBox("Process Status")
        status_layout = QVBoxLayout(gb_status)
        
        self.lbl_pcm_status = QLabel("⚪ No File Loaded")
        self.lbl_pcm_status.setStyleSheet("color: #757575; font-weight: bold;")
        self.lbl_pcm_status.setWordWrap(True)
        status_layout.addWidget(self.lbl_pcm_status)
        
        self.lbl_refine_status = QLabel("🔳 View: Low-Res")
        self.lbl_refine_status.setStyleSheet("color: #9E9E9E; font-weight: bold;")
        self.lbl_refine_status.setWordWrap(True)
        status_layout.addWidget(self.lbl_refine_status)
        
        self.lbl_upgrade_status = QLabel("💤 Master HD: Idle")
        self.lbl_upgrade_status.setStyleSheet("color: #BDBDBD; font-weight: bold;")
        self.lbl_upgrade_status.setWordWrap(True)
        status_layout.addWidget(self.lbl_upgrade_status)
        
        dock_layout.addWidget(gb_status)
        
        # E. AI Expert Operator (Silent Discovery)
        gb_aibot = QGroupBox("AI Expert Operator")
        aibot_layout = QVBoxLayout(gb_aibot)
        
        # v30.2: Feature Icon in Label
        icon_row = QHBoxLayout()
        self.lbl_aibot_icon = QLabel()
        if os.path.exists(AI_ICON_PATH):
            self.lbl_aibot_icon.setPixmap(QPixmap(AI_ICON_PATH).scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        icon_row.addWidget(self.lbl_aibot_icon)
        icon_row.addWidget(QLabel("<b>Expert System Active</b>"))
        icon_row.addStretch()
        aibot_layout.addLayout(icon_row)
        
        self.btn_enable_aibot = QPushButton("🤖 Enable AI Bot")
        self.btn_enable_aibot.setCheckable(True)
        self.btn_enable_aibot.setStyleSheet("font-weight: bold; padding: 5px; color: #673AB7;")
        self.btn_enable_aibot.toggled.connect(self._on_aibot_toggled)
        
        # v35.5: Compact Grid Layout for AI Controls
        grid_aibot = QGridLayout()
        grid_aibot.addWidget(self.btn_enable_aibot, 0, 0)
        
        # YAMNet Auto-Label
        self.btn_yamnet = QPushButton("🧠 YAMNet")
        self.btn_yamnet.setToolTip("Auto-Label everything using standard YAMNet audio database.")
        self.btn_yamnet.setEnabled(False)
        self.btn_yamnet.clicked.connect(self._on_aibot_yamnet_clicked)
        self.btn_yamnet.setStyleSheet("font-weight: bold; background-color: #E8F5E9; color: #2E7D32;")
        grid_aibot.addWidget(self.btn_yamnet, 0, 1)
        
        # Expert Training Button
        self.btn_train_expert = QPushButton("🎓 Teach AI")
        self.btn_train_expert.setEnabled(False)
        self.btn_train_expert.setToolTip("Learn from your manual labels.")
        self.btn_train_expert.setStyleSheet("background-color: #F3E5F5; color: #7B1FA2; font-weight: bold;")
        self.btn_train_expert.clicked.connect(self._on_aibot_train_expert_clicked)
        grid_aibot.addWidget(self.btn_train_expert, 1, 0)
        
        # Controls for results
        self.btn_view_results = QPushButton("📋 Results")
        self.btn_view_results.setEnabled(False)
        self.btn_view_results.clicked.connect(self._on_aibot_view_results_clicked)
        grid_aibot.addWidget(self.btn_view_results, 1, 1)

        aibot_layout.addLayout(grid_aibot)

        # Minor utility buttons
        utils_row = QHBoxLayout()
        self.btn_clear_overlays = QPushButton("🗑️ Clear Overlays")
        self.btn_clear_overlays.setStyleSheet("color: #B71C1C; font-size: 10px;")
        self.btn_clear_overlays.clicked.connect(self._clear_ghost_spans)
        utils_row.addWidget(self.btn_clear_overlays)

        self.btn_reset_expert = QPushButton("⚙️ Reset Expert")
        self.btn_reset_expert.setStyleSheet("color: #757575; font-size: 10px;")
        self.btn_reset_expert.clicked.connect(self._on_aibot_reset_expert_clicked)
        utils_row.addWidget(self.btn_reset_expert)
        
        aibot_layout.addLayout(utils_row)

        
        dock_layout.addWidget(gb_aibot)
        
        dock_layout.addStretch()
        # self.dock.setWidget(dock_content)  <-- v33.0: Removed! Scroll area is already set in init_ui.
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        # dict: channel_name -> list of Line2D objects (one per channel)
        self.wave_lines = {}

        # v32.14: Resource Meter in Status Bar
        self.lbl_resource_status = QLabel("[RAM: --%] [CPU: --%]")
        self.lbl_resource_status.setStyleSheet("color: #4CAF50; font-weight: bold; margin-right: 10px;")
        self.lbl_resource_status.setToolTip("System Resources (Available Memory & CPU Load)")
        self.statusBar().addPermanentWidget(self.lbl_resource_status)

    # ─────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────
    def get_color_for_class(self, label):
        # v36.8: Normalize label (strip ⭐ and spaces) so AI labels match manual ones
        clean_label = label.replace("⭐", "").strip()
        key = clean_label.lower()
        
        if key not in self.class_colors:
            # Map currently used colors to normalized hex to skip them
            used_hex = {mcolors.to_hex(c) for c in self.class_colors.values()}
            
            picked = None
            # Scan our available palette for anything not already 'taken'
            for _ in range(len(self.available_colors)):
                trial = self.available_colors[self.color_index % len(self.available_colors)]
                self.color_index += 1
                if mcolors.to_hex(trial) not in used_hex:
                    picked = trial
                    break
            
            if picked is None:
                # Fallback: all colors used, cycle back to the beginning
                picked = self.available_colors[self.color_index % len(self.available_colors)]
                self.color_index += 1
                
            self.class_colors[key] = picked
        return self.class_colors[key]

    def get_base_datetime(self):
        dt_str = self.datetime_entry.text().strip()
        try:
            if "." in dt_str:
                return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            QMessageBox.critical(self, "Format Error", "Invalid Datetime format.")
            return None

    def get_default_win_dur(self):
        """v32.15: Hardware-aware starting default for initial file load."""
        if self.sample_rate and self.sample_rate >= 44100:
            return 100.0 # Dense 48kHz audio starts smaller
        return 300.0     # 16kHz AMMS starts at standard 5-min view

    def get_max_win_dur(self):
        """v32.15: Safety limit for window expansion."""
        # Relaxed Cap: User allowed up to 300s regardless of sample rate
        rate_limit = 300.0
            
        # Physical RAM safe-mode override
        _, ram_avail, _ = self.latest_resources
        if ram_avail < 1.0: # < 1GB
            return 60.0 # Critical Safe Mode
            
        return rate_limit

    def get_total_duration(self):
        """Returns the length of the file in seconds, prioritizing cached metadata if PCM is still buffering."""
        if hasattr(self, 'total_duration_v9'):
            return self.total_duration_v9
        if self.audio_data is not None and self.sample_rate:
            return self.audio_data.shape[0] / self.sample_rate
        return 0.0

    # ─────────────────────────────────────────────────────────
    #  Channel checkbox visibility
    # ─────────────────────────────────────────────────────────
    def update_channel_visibility(self):
        for ch in CH_INFO:
            name    = ch["name"]
            visible = self.ch_checkboxes[name].isChecked()
            for line in self.wave_lines.get(name, []):
                line.set_visible(visible)
        self.update_background()

    # ─────────────────────────────────────────────────────────
    #  Load WAV
    # ─────────────────────────────────────────────────────────
    def load_wav(self, overriding_filepath=None):
        if overriding_filepath:
            filepath = overriding_filepath
        else:
            filepath, _ = QFileDialog.getOpenFileName(self, "Load WAV File", "", "WAV files (*.wav)")
            
        if not filepath:
            return

        # Explicit Multi-Window support check natively
        if self.audio_data is not None and not overriding_filepath:
            new_window = AudioAnnotatorApp()
            global_app_instances.append(new_window)
            new_window.showMaximized()
            # Feed the exact selected filepath into the spawned instance securely
            new_window.load_wav(overriding_filepath=filepath)
            return

        # v24.3: Attempt to extract recording time from filename (e.g. AMMS_20150303_204457.wav)
        fname = os.path.basename(filepath)
        pattern = r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})"
        match = re.search(pattern, fname)
        if match:
            extracted = f"{match.group(1)}-{match.group(2)}-{match.group(3)} {match.group(4)}:{match.group(5)}:{match.group(6)}"
            self.datetime_entry.setText(extracted)

        # Setup Progress GUI Modal Wrapper preventing UI freezes
        self.progress_dialog = QProgressDialog("Initializing loading sequence...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle(f"Loading {os.path.basename(filepath)}")
        self.progress_dialog.setWindowIcon(QIcon(ICON_PATH))
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setValue(0)
        
        # 1. v31.2: Auto-detect local AMMS Log for Time Sync (Highest Priority)
        # User requirement: exactly same name as wav + .LOG
        log_candidate = os.path.splitext(filepath)[0] + ".LOG"
        self.amms_timing = None
        log_calibrated  = False
        self._pending_calibration_msg = None  # v31.3: Delay message until load finish
        
        if os.path.exists(log_candidate):
            print(f"Auto-Sync: Found local log file: {log_candidate}")
            # v32.0: Get WAV Rate ahead of time for sync scaling
            try:
                wav_sr = sf.info(filepath).samplerate
            except:
                wav_sr = 16000
                
            self.amms_timing = amms_timing.parse_amms_log(log_candidate, wav_sr=wav_sr)
            if self.amms_timing:
                start_dt = self.amms_timing.sample2time(0)
                if start_dt:
                    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                    self.datetime_entry.setText(start_str)
                    self._on_base_datetime_changed()
                    
                    # v31.2: Capture calibration source message for later
                    meta = self.amms_timing.get_metadata_for_sample(0)
                    source = meta.get('_source', 'Unknown') if meta else 'Unknown'
                    self._pending_calibration_msg = (
                        f"<b>Log Auto-Calibration Successful</b><br><br>"
                        f"<b>Start Time:</b> {start_str}<br>"
                        f"<b>Time Source:</b> <font color='blue'>{source}</font><br><br>"
                        f"This timestamp has been retrieved directly from the .LOG hardware clock."
                    )
                    
                    print(f"Auto-Sync: Calibrated start time from LOCAL LOG ({source}): {start_str}")
                    log_calibrated = True

        # 2. v25.2: Master Log Search (Excel) - Only if local log didn't already calibrate
        if not log_calibrated:
            sensor_id = self.extract_sensor_id(filepath)
            if sensor_id:
                official_start = self.fetch_timing_from_master_log(sensor_id, wav_dir=os.path.dirname(filepath))
                if official_start:
                    self.datetime_entry.setText(official_start)
                    self._on_base_datetime_changed()
                    print(f"Auto-Sync: Calibrated from Master Excel: {official_start}")

        # Start loaders
        self.loader_thread = AudioLoaderThread(filepath)
        self.loader_thread.progress.connect(self._on_load_progress)
        self.loader_thread.finished_success.connect(self._on_load_success)
        self.loader_thread.error.connect(self._on_load_error)
        self.progress_dialog.canceled.connect(self._on_load_canceled)
        
        self.progress_dialog.show()
        self.loader_thread.start()

    def _on_load_progress(self, value, text):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(text)
            
    def _on_load_canceled(self):
        if hasattr(self, 'loader_thread') and self.loader_thread.isRunning():
            self.loader_thread.terminate()
            self.loader_thread.wait()
            
    def _on_load_error(self, err_msg):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        QMessageBox.critical(self, "Thread Error", f"Fatal background calculation failure:\n{err_msg}")
        
    def _on_load_success(self, payload):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            
        # v31.3: Show delayed calibration message
        if getattr(self, '_pending_calibration_msg', None):
            QMessageBox.information(self, "AMMS Calibration", self._pending_calibration_msg)
            self._pending_calibration_msg = None
        
        # v32.96: Always refresh the tree to ensure the absolute timeline is projected
        self.refresh_tree_display()
            
        self.sample_rate = payload["sr"]
        self.n_channels = payload["n_channels"]
        self.filepath = payload["filepath"] # Store for refinement
        self.total_duration_v9 = payload["duration"]
        self.annotation_file = self.filepath.replace(".wav", "_annotations.csv")
        
        # v9 Upgrade: If loaded from cache, the raw PCM data is still on disk.
        # Spawn the background streamer to load it for playback.
        if payload.get("from_cache") or payload["data"] is None:
            self.audio_data = None
            self.is_pcm_buffering = True
            self.lbl_pcm_status.setText("⏳ Audio: Buffering...")
            self.lbl_pcm_status.setStyleSheet("color: #FF9800; font-weight: bold;")
            self.pcm_streamer = AudioPCMStreamer(payload["filepath"])
            self.pcm_streamer.finished.connect(self._on_pcm_ready)
            self.pcm_streamer.start()
        else:
            self.audio_data = payload["data"]
            self.is_pcm_buffering = False
            self.lbl_pcm_status.setText("🔊 Audio: Ready")
            self.lbl_pcm_status.setStyleSheet("color: #4CAF50; font-weight: bold;")

        self.is_master_hd = payload.get("is_master_hd", False)
        
        # v32.15: Initialize window fields using smart starting defaults
        self.entry_start_time.setText(f"{0.0:.6f}")
        start_dur = min(self.get_default_win_dur(), self.get_total_duration())
        self.entry_end_time.setText(f"{start_dur:.6f}")
        self.entry_duration.setText(f"{start_dur:.6f}")

        # v26: Automatically scale amplitude to fit the frame on load
        self.auto_scale_amplitude()
        
        # v16 Data Decoupling Logic:
        # If the file is already Master HD, the 'rgb_base' is huge (50k+ cols).
        # We don't want to imshow the huge matrix directly across the whole file.
        # Instead, we store it in master_hd_data and create a fast 8000-col 'Overview' proxy.
        if self.is_master_hd:
            self.lbl_upgrade_status.setText("🔊 Master HD: Ready (Cached)")
            self.lbl_upgrade_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            # Store the master data
            self.master_hd_data = {
                "rgb": payload["rgb_base"],
                "t": payload["t_spec"],
                "f": payload["f_bins"]
            }
            
            # Create a lightweight Overview proxy (8000 cols) for fast initial shifts
            rgb_full = payload["rgb_base"]
            n_cols = rgb_full.shape[1]
            if n_cols > MAX_SPEC_COLS:
                step = n_cols // MAX_SPEC_COLS
                payload["rgb_base"] = rgb_full[:, ::step, :]
                payload["t_spec"]   = payload["t_spec"][::step]
        
        self._rebuild_plots_fast(payload)
        self._load_annotations()
        
        # v14 Upgrade: If not already Master HD, start background refiner
        if not self.is_master_hd:
            self.lbl_upgrade_status.setText("🚀 Master HD: Starting...")
            self.lbl_upgrade_status.setStyleSheet("color: #9C27B0; font-weight: bold;")
            self.master_hd_refiner = FullFileHDRefiner(self.filepath, self.sample_rate, self.n_channels)
            self.master_hd_refiner.progress.connect(self._on_master_hd_progress)
            self.master_hd_refiner.finished.connect(self._on_master_hd_finished)
            self.master_hd_refiner.start()
        else:
            self.lbl_refine_status.setText("🔳 View: HD (Master)")
            self.lbl_refine_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            # Force immediate HD slice extraction for the first window
            self.trigger_refinement()

    def _on_pcm_ready(self, data):
        """Callback when the background 2GB+ read finishes."""
        self.audio_data = data
        self.is_pcm_buffering = False
        self.lbl_pcm_status.setText("🔊 Audio: Ready")
        self.lbl_pcm_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        
        # v26.4: Auto-scale once the data is actually in RAM
        self.auto_scale_amplitude()

    # ─────────────────────────────────────────────────────────
    #  Interactivity Handlers
    # ─────────────────────────────────────────────────────────
    def on_select_interval(self, xmin, xmax):
        """Callback from SpanSelector (both waveform and spectrogram)."""
        # v36.0: Ignore if user is holding Ctrl (Play/Zoom mode)
        modifiers = QApplication.keyboardModifiers()
        if bool(modifiers & Qt.ControlModifier):
            return

        # Ignore micro-clicks (less than 10ms)
        if abs(xmax - xmin) < 0.01: 
            return
        
        # Show dialog to get class and weight
        dlg = AnnotationEditDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self._add_annotation_to_tree(xmin, xmax, dlg.class_label, 
                                         weight=dlg.weight, autosave=True)

    # Redundant handlers (on_click, on_scroll) removed. Combined versions exist at the bottom of the class.

    @staticmethod
    def _complex_intensity_rgb(data, sr, progress_cb=None, force_max_cols=None):
        """
        v30.8: Chunked-STFT Processing.
        Prevents massive RAM allocation failures by processing audio in segments.
        """
        if progress_cb: progress_cb(5, "Initializing Chunked Spectral Engine...")
        
        n_samples = data.shape[0]
        actual_max_cols = force_max_cols if force_max_cols else MAX_SPEC_COLS
        target_cols = actual_max_cols * 2 # HD Multiplier
        
        # 1. Determine optimal STFT overlap
        required_step = max(1, n_samples // target_cols)
        adaptive_overlap = max(0, SPEC_NFFT - required_step)
        final_overlap = min(SPEC_OVERLAP, int(adaptive_overlap))
        step_size = SPEC_NFFT - final_overlap
        
        # 2. Pre-calculate total spectral dimensions
        f_bins = np.fft.rfftfreq(SPEC_NFFT, 1 / sr)
        n_freqs = len(f_bins)
        
        # Calculate exactly how many STFT columns scipy would produce
        total_stft_cols = max(1, (n_samples - final_overlap) // step_size)
        downsample_step = max(1, total_stft_cols // target_cols)
        final_cols = total_stft_cols // downsample_step
        
        # 3. Pre-allocate final output buffers (small footprint)
        full_rgb = np.zeros((n_freqs, final_cols, 3), dtype=np.float32)
        full_w_s = np.zeros((n_freqs, final_cols), dtype=np.float32)
        full_t   = np.zeros(final_cols, dtype=np.float32)
        
        # 4. Segmented Processing Loop
        # We process ~10,000 columns at a time (approx 160MB peak RAM)
        chunk_cols = 10000
        n_chunks = (total_stft_cols + chunk_cols - 1) // chunk_cols
        
        # Prepare STFT parameters
        win = np.hamming(SPEC_NFFT).astype(np.float32)
        kw  = dict(fs=sr, window=win, nperseg=SPEC_NFFT,
                   noverlap=final_overlap, return_onesided=True)
        
        processed_stft_cols = 0
        final_cols_written = 0
        
        for i in range(n_chunks):
            start_col = i * chunk_cols
            end_col   = min(start_col + chunk_cols, total_stft_cols)
            cols_to_process = end_col - start_col
            
            if progress_cb:
                prog = 10 + int(70 * (i / n_chunks))
                progress_cb(prog, f"Processing Segment {i+1}/{n_chunks}...")
            
            # Calculate exactly which samples we need for this column range
            # scipy.signal.stft uses: i*step to i*step + nperseg
            start_sample = start_col * step_size
            end_sample   = (end_col - 1) * step_size + SPEC_NFFT
            
            # Slice and Normalize (float32 for RAM safety)
            # v32.0: Force float32 divisor to prevent intermediate 64-bit promotion
            chunk_pcm = (data[start_sample:end_sample] / np.float32(32768.0)).astype(np.float32)
            
            # STFT for 3 channels
            _, tc, P  = stft(chunk_pcm[:, 0], **kw)
            _, _,  Vx = stft(chunk_pcm[:, 1], **kw)
            _, _,  Vy = stft(chunk_pcm[:, 2], **kw)
            
            # Ensure complex64 for peak reduction
            P  = P.astype(np.complex64)
            Vx = Vx.astype(np.complex64)
            Vy = Vy.astype(np.complex64)
            
            # DOA / Intensity Math
            I = (Vx * np.conj(P)).real + 1j * (Vy * np.conj(P)).real
            del P, Vx, Vy # Immediate flush
            
            A = (1j * I).astype(np.complex64)
            H = ((np.angle(A) + np.pi).astype(np.float32) / (2.0 * np.pi)).astype(np.float32)
            mag = np.abs(A).astype(np.float32)
            n_val = mag.max() if mag.max() > 0 else 1.0
            w = np.log(np.sqrt(mag ** 2 + (n_val * 1e-8) ** 2).astype(np.float32)).astype(np.float32)
            
            w_min, w_max = min(w.min(), 0.0), max(w.max(), 0.0)
            w_s_chunk = np.ones_like(w, dtype=np.float32)
            if w_max != w_min:
                w_s_chunk = ((w - w_min) / (w_max - w_min)).astype(np.float32)
            
            S = np.cos(0.6 * w_s_chunk * np.pi / 2.0).astype(np.float32)
            V = np.sin(0.6 * w_s_chunk * np.pi / 2.0).astype(np.float32)
            del I, A, mag, w
            
            # Downsample this chunk specifically
            n_chunk_cols = H.shape[1]
            # Offset tracking: ensure we pick indices aligned with global downsample_step
            # We want to pick global indices: g = i * step_size
            # Local index j in chunk corresponds to global index start_col + j
            indices = np.where((np.arange(start_col, end_col) % downsample_step) == 0)[0]
            
            if len(indices) > 0:
                h_sub = H[:, indices]
                s_sub = S[:, indices]
                v_sub = V[:, indices]
                w_sub = w_s_chunk[:, indices]
                t_sub = (tc[indices] + (start_sample / sr))
                
                # Stack to RGB
                hsv_sub = np.stack([h_sub, s_sub, v_sub], axis=-1)
                rgb_sub = mcolors.hsv_to_rgb(hsv_sub).astype(np.float32)
                
                # Write to pre-allocated buffers
                n_to_write = min(len(indices), final_cols - final_cols_written)
                if n_to_write > 0:
                    full_rgb[:, final_cols_written : final_cols_written + n_to_write, :] = rgb_sub[:, :n_to_write, :]
                    full_w_s[:, final_cols_written : final_cols_written + n_to_write] = w_sub[:, :n_to_write]
                    full_t[final_cols_written : final_cols_written + n_to_write] = t_sub[:n_to_write]
                    final_cols_written += n_to_write
            
            del H, S, V, w_s_chunk, tc # Flush chunk RAM
            
        if progress_cb: progress_cb(100, "Segmented Spectral Analysis Complete.")
        return full_rgb, full_w_s, f_bins, full_t

    def _rebuild_plots_fast(self, payload):
        """Asynchronously receive thread math directly to hardware canvases."""
        filepath = payload["filepath"]
        data = payload["data"] # Full PCM (may be None if streaming)
        env  = payload.get("wave_envelope") # Decimated Envelope (always available)
        sr = payload["sr"]
        n_ch = payload["n_channels"]
        duration = payload["duration"]
        
        self.stop_audio()
        self.play_marker  = None
        self.play_text    = None
        self.spec_marker  = None
        self.wave_lines   = {}
        self.spans.clear()
        
        for s in getattr(self, 'ghost_spans', []):
            try: s[0].remove()
            except: pass
            try: s[1].remove()
            except: pass
        if hasattr(self, 'ghost_spans'):
            self.ghost_spans.clear()
        self.tree.clear()
        self.annotations  = []
        
        # Reset Image cache trackers seamlessly avoiding matrix leakage!
        self._spec_rgb_base = payload.get("rgb_base")
        self._spec_intensity_base = payload.get("intensity_base")
        self._spec_mono_base = payload.get("mono_base")
        self._spec_img_obj = None
        self._spec_t_bins = payload.get("t_spec")
        self._spec_f_bins = payload.get("f_bins")
        
        try:
            self.slider_contrast.blockSignals(True)
            self.slider_brightness.blockSignals(True)
            self.slider_contrast.setValue(100)
            self.slider_brightness.setValue(0)
            self.slider_contrast.blockSignals(False)
            self.slider_brightness.blockSignals(False)
        except: pass

        # ── Waveform axes ────────────────────────────────────
        self.ax.clear()
        self.ax.set_title(
            f"Waveform: {os.path.basename(filepath)}  "
            f"({n_ch} ch · {sr} Hz)  "
            "— Ctrl+Click to play · drag to annotate", pad=40)
        self.ax.set_ylabel("Amplitude")
        plt.setp(self.ax.get_xticklabels(), visible=False)

        # PLOT LOGIC: Use 'env' if 'data' is still buffering
        source_data = data if data is not None else env
        if source_data is None:
            # We have absolutely no visual data for the waveform (should be rare with the cache-hit guard)
            self.ax.text(0.5, 0.5, "Waveform data still loading from disk...", 
                         ha='center', va='center', transform=self.ax.transAxes, clip_on=True)
            self.canvas.draw_idle()
            return

        max_pts = 300_000
        
        for ch in CH_INFO:
            idx  = ch["idx"]
            name = ch["name"]
            if idx >= n_ch:
                self.wave_lines[name] = []
                continue
            
            col_data = source_data[:, idx]
            if len(col_data) > max_pts:
                step    = len(col_data) // max_pts
                pd_     = col_data[::step]
                t_axis  = np.arange(0, len(col_data), step) / sr
            else:
                pd_     = col_data
                # If using envelope, the scale is different
                if data is None:
                    # Envelopes are pre-downsampled to 100k pts over the full duration
                    t_axis = np.linspace(0, duration, len(col_data))
                else:
                    t_axis = np.arange(len(col_data)) / sr

            (line,) = self.ax.plot(
                t_axis, pd_,
                color=ch["color"], linewidth=0.5, alpha=0.85,
                label=name, visible=self.ch_checkboxes[name].isChecked()
            )
            self.wave_lines[name] = [line]

        if any(n < n_ch for n in [ch["idx"] for ch in CH_INFO if ch["idx"] < n_ch]):
            self.ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
            
        # Programmatically wire robust Default Y Limits
        y_min = np.min(source_data) if source_data is not None else -1.0
        y_max = np.max(source_data) if source_data is not None else 1.0
        margin_y = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        
        self.ax.set_ylim(y_min - margin_y, y_max + margin_y)
        self.entry_amp_min.setText(f"{y_min - margin_y:.2f}")
        self.entry_amp_max.setText(f"{y_max + margin_y:.2f}")

        # Sync the window span completely to the configured X Duration settings!
        try:
            init_duration_window = min(float(self.entry_duration.text()), duration)
        except:
            init_duration_window = 300.0
        self.entry_start_time.setText(f"{0.0:.6f}")
        self.ax.set_xlim(0, init_duration_window)

        # ── Spectrogram / TimeFreqAngle axes ─────────────────
        self.ax_spec.clear()
        self.ax_spec.set_xlabel("Time (seconds)")
        self.ax_spec.set_ylabel("Frequency (Hz)")

        # v28.1: Unified Spectral Tracker
        self._spec_active_rgb       = payload.get("rgb_base")
        self._spec_active_intensity = payload.get("intensity_base")
        self._spec_active_mono      = payload.get("mono_base")
        ts                          = payload.get("t_spec")
        fs                          = payload.get("f_bins") if payload.get("f_bins") is not None else payload.get("mono_f_bins")
        
        if n_ch >= 3 and self._spec_active_rgb is not None:
            self.ax_spec.set_title("Time-Frequency-Angle [Hue=Bearing · Brightness=Intensity]", pad=15)
            self._spec_img_obj = self.ax_spec.imshow(
                self._spec_active_rgb, aspect='auto', origin='lower',
                extent=[ts[0], ts[-1], fs[0], fs[-1]], interpolation='nearest'
            )
        elif self._spec_active_mono is not None:
            self.ax_spec.set_title("Spectrogram (Power Density)", pad=15)
            ts_mono = [0, duration]
            self._spec_img_obj = self.ax_spec.imshow(
                self._spec_active_mono, aspect='auto', origin='lower',
                extent=[ts_mono[0], ts_mono[-1], fs[0], fs[-1]], cmap='magma', interpolation='nearest'
            )
        
        # v28.1: Apply current brightness/gamma settings immediately to the initial plot
        self.update_spec_appearance()

        # v32.13: Default to 8kHz max for consistent viewing across hardware rates
        try:
            custom_min = float(self.entry_freq_min.text())
            custom_max = float(self.entry_freq_max.text())
            self.ax_spec.set_ylim(custom_min, custom_max)
        except:
            default_max = min(8000, sr / 2)
            self.ax_spec.set_ylim(0, default_max)
            self.entry_freq_min.setText(f"{0.0:.2f}")
            self.entry_freq_max.setText(f"{default_max:.2f}")

        self.ax_spec.set_xlim(0, init_duration_window)

        # ── SpanSelectors on both axes ────────────────────────
        self.span_selector = SpanSelector(
            self.ax, self.on_select_interval, 'horizontal', useblit=True,
            props=dict(alpha=0.3, facecolor='green'), interactive=False
        )
        self.span_selector_spec = SpanSelector(
            self.ax_spec, self.on_select_interval, 'horizontal', useblit=True,
            props=dict(alpha=0.3, facecolor='green'), interactive=False
        )

        # ── Animated playhead markers ────────────────────────
        self.play_marker = self.ax.axvspan(
            0, 0, color='#FFD700', alpha=0.35, animated=True)
        self.play_marker.set_visible(False)

        self.play_text = self.ax.text(
            0, 0, "", color='#DAA520', weight='bold',
            horizontalalignment='center', backgroundcolor='white',
            animated=True, zorder=10, clip_on=False)
        self.play_text.set_visible(False)

        self.spec_marker = self.ax_spec.axvline(
            x=0, color='yellow', linewidth=1.5, animated=True)
        self.spec_marker.set_visible(False)
        
        self.play_text_sec = self.ax_spec.text(
            0, 0, "", color='yellow', weight='bold',
            horizontalalignment='center', backgroundcolor='black',
            animated=True, zorder=10, clip_on=False)
        self.play_text_sec.set_visible(False)

        self.update_background()

    @staticmethod
    def _draw_power_spectrogram_core(p_channel, sr, duration, progress_cb=None):
        """v26.7: Added normalization (base 32768) for raw sensor scaling stability."""
        # Normalize to standard [-1, 1] range for spectral engine stability
        normalized_data = p_channel / 32768.0
        
        if progress_cb: progress_cb(20, "Executing STFT for Power density...")
        f_bins, t_spec, Sxx = spectrogram(
            normalized_data, fs=sr, window='hamming',
            nperseg=SPEC_NFFT, noverlap=SPEC_OVERLAP, scaling='spectrum'
        )
        if progress_cb: progress_cb(60, "Normalizing Log-Peak spectrum...")
        Sxx_max = np.max(Sxx) if np.max(Sxx) > 0 else 1.0
        Sxx_dB  = 10 * np.log10(Sxx / Sxx_max + 1e-8)
        if progress_cb: progress_cb(90, "Raster optimization...")
        return f_bins, t_spec, Sxx_dB

    # ─────────────────────────────────────────────────────────
    #  Dock Widget Engine Overrides
    # ─────────────────────────────────────────────────────────
    def _on_window_start_changed(self):
        """When Start moves, move End to maintain Duration, capped at adaptive total."""
        try:
            start = float(self.entry_start_time.text())
            dur = float(self.entry_duration.text())
            max_dur = self.get_max_win_dur()
            dur = min(max_dur, max(0.1, dur))
            end = start + dur
            
            self.entry_duration.setText(f"{dur:.3f}")
            self.entry_end_time.setText(f"{end:.3f}")
        except: pass

    def _on_window_end_changed(self):
        """When End moves, update Duration, capped at adaptive gap from Start."""
        try:
            start = float(self.entry_start_time.text())
            end = float(self.entry_end_time.text())
            max_dur = self.get_max_win_dur()
            
            # Constraint: Gap must be <= max_dur
            if (end - start) > max_dur:
                end = start + max_dur
                self.entry_end_time.setText(f"{end:.3f}")
            
            # Constraint: Duration can't be negative
            if end <= start:
                end = start + 0.1
                self.entry_end_time.setText(f"{end:.6f}")
            
            self.entry_duration.setText(f"{end - start:.6f}")
        except: pass

    def _on_window_dur_changed(self):
        """When Duration moves, update End, capped at adaptive max."""
        try:
            start = float(self.entry_start_time.text())
            dur = float(self.entry_duration.text())
            max_dur = self.get_max_win_dur()
            
            if dur > max_dur:
                dur = max_dur
                self.entry_duration.setText(f"{max_dur:.3f}")
            
            if dur <= 0:
                dur = 0.1
                self.entry_duration.setText(f"{0.1:.6f}")
                
            self.entry_end_time.setText(f"{start + dur:.6f}")
        except: pass

    def apply_time_window(self):
        dur_total = self.get_total_duration()
        if dur_total <= 0: return
        
        try:
            start_t = float(self.entry_start_time.text())
            end_t = float(self.entry_end_time.text())
            
            # v10 Safety: Cap visible window
            max_dur = self.get_max_win_dur()
            dur = end_t - start_t
            if dur > max_dur:
                dur = max_dur
                end_t = start_t + dur
                self.entry_end_time.setText(f"{end_t:.6f}")
                self.entry_duration.setText(f"{max_dur:.6f}")
                self.lbl_pcm_status.setText(f"⚠️ Capped window to {max_dur}s")
                self.lbl_pcm_status.setStyleSheet("color: #E91E63; font-weight: bold;")
            else:
                # v24.2: Revert to 'Ready' status if the window is no longer capped
                if self.audio_data is not None:
                    self.lbl_pcm_status.setText("🔊 Audio: Ready")
                    self.lbl_pcm_status.setStyleSheet("color: #4CAF50; font-weight: bold;")

            # Clamp to file bounds
            start_t = max(0, min(start_t, dur_total))
            end_t = max(start_t + 0.1, min(end_t, dur_total))
            
            self.ax.set_xlim(start_t, end_t)
            
            # Update fields in case they were clamped
            self.entry_start_time.setText(f"{start_t:.6f}")
            self.entry_end_time.setText(f"{end_t:.6f}")
            self.entry_duration.setText(f"{end_t - start_t:.6f}")
            
            # v16.2: Reset active trackers to overview during the transition
            # This ensures we don't show a 'blank' screen or the 'old' window 
            # while waiting for the refinement slice.
            if hasattr(self, '_spec_rgb_base'):
                self._spec_active_rgb = self._spec_rgb_base
                self._spec_img_obj.set_extent([0, dur_total, self.ax_spec.get_ylim()[0], self.ax_spec.get_ylim()[1]])
            elif hasattr(self, '_spec_mono_base'):
                self._spec_active_mono = self._spec_mono_base
                self._spec_img_obj.set_extent([0, dur_total, self.ax_spec.get_ylim()[0], self.ax_spec.get_ylim()[1]])

            # v24: Every time we shift the window, we discard the 'Ultra-HD' refined view 
            # and go back to the instant background overview.
            self.is_view_refined = False
            self.lbl_refine_status.setText("🔳 View: Ready")
            self.lbl_refine_status.setStyleSheet("color: #9E9E9E; font-weight: bold;")

            self.update_background()
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", "Please parse numerical time frames correctly!")

    def shift_window_next(self):
        dur_total = self.get_total_duration()
        if dur_total <= 0: return
        try:
            start_t = float(self.entry_start_time.text())
            dur = float(self.entry_duration.text())
            
            new_start = start_t + dur
            if new_start >= dur_total: return
            new_end = new_start + dur
            
            self.entry_start_time.setText(f"{new_start:.6f}")
            self.entry_end_time.setText(f"{new_end:.6f}")
            self.apply_time_window()
        except ValueError:
            pass

    def shift_window_prev(self):
        if self.get_total_duration() <= 0: return
        try:
            start_t = float(self.entry_start_time.text())
            dur = float(self.entry_duration.text())
            
            new_start = max(0.0, start_t - dur)
            new_end = new_start + dur
            
            self.entry_start_time.setText(f"{new_start:.6f}")
            self.entry_end_time.setText(f"{new_end:.6f}")
            self.apply_time_window()
        except ValueError:
            pass
            
    def _on_base_datetime_changed(self):
        """v24.3: Refresh everything when the user overrides the main recording start time."""
        self.refresh_tree_display()
        self.update_background()

    def refresh_tree_display(self):
        """v25.1: Re-calculate all visible date strings in the tree based on new base_dt."""
        base_dt = self.get_base_datetime()
        if not base_dt: return
        
        for ann in self.annotations:
            xmin = ann['xmin']
            xmax = ann['xmax']
            
            if self.amms_timing:
                dt_s = self.amms_timing.sample2time(xmin * self.sample_rate)
                dt_e = self.amms_timing.sample2time(xmax * self.sample_rate)
                start_str = dt_s.strftime("%Y-%m-%d %H:%M:%S.%f")
                end_str   = dt_e.strftime("%Y-%m-%d %H:%M:%S.%f")
            else:
                start_str = (base_dt + timedelta(seconds=xmin)).strftime("%Y-%m-%d %H:%M:%S.%f")
                end_str   = (base_dt + timedelta(seconds=xmax)).strftime("%Y-%m-%d %H:%M:%S.%f")
            
            item = ann['item']
            item.setText(3, start_str)
            item.setText(4, end_str)
            
            # v32.68: If we have amms_timing, ensure samples are also correct in case rate shifted
            if self.amms_timing:
                item.setText(1, str(int(xmin * self.sample_rate + self.amms_timing.audio_start_sample)))
                item.setText(2, str(int(xmax * self.sample_rate + self.amms_timing.audio_start_sample)))

    def extract_sensor_id(self, filepath):
        """v25: Robustly extract sensor IDs like 'AMMS102' from the path or filename."""
        fname = os.path.basename(filepath).upper()
        # Search for AMMS followed by digits
        match = re.search(r"(AMMS\d+)", fname)
        if match:
            return match.group(1)
        
        # Fallback: Check parent directory if it's nested (e.g. Q:\...\AMMS102\AMMS-001.wav)
        pdir = os.path.basename(os.path.dirname(filepath)).upper()
        match = re.search(r"(AMMS\s*\d+)", pdir)
        if match:
            # Normalize by removing space: AMMS 114 -> AMMS114
            return match.group(1).replace(" ", "")
            
        return None

    def fetch_timing_from_master_log(self, sensor_id, wav_dir=None):
        """v25.2: Automated lookup with progressive search (Local -> Q: drive)."""
        # Search Location 1: Local WAV folder
        paths_to_try = []
        if wav_dir:
            paths_to_try.append(os.path.join(wav_dir, 'Bavaria_Drones_Timestamps.xlsx'))
            
        # Search Location 2: Official Master Repository on Q:
        paths_to_try.append(r'Q:\2015\2015_03_03_ESG Bavaria Drones\Raw data\Bavaria_Drones_Timestamps.xlsx')
        
        for master_log_path in paths_to_try:
            try:
                if not os.path.exists(master_log_path):
                    continue

                df = pd.read_excel(master_log_path)
                # Normalize Excel sensor names for robust matching
                df['SensorSync'] = df['Sensor'].astype(str).str.upper().str.replace(" ", "")
                
                # Find matching row
                row = df[df['SensorSync'] == sensor_id.upper()]
                if not row.empty:
                    # Get the first match
                    start_dt = row.iloc[0]['Start Date Time']
                    if pd.notnull(start_dt):
                        # Convert to string format for the UI
                        if hasattr(start_dt, 'strftime'):
                            return start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        return str(start_dt)
            except Exception as e:
                print(f"Lookup error at {master_log_path}: {e}")
        
        return None

        # Note: We don't trigger apply_time_window here to avoid jitter; 
        # user must click 'Apply Time View' or change duration to commit.

    def apply_amp_limits(self):
        if self.audio_data is None: return
        try:
            min_a = float(self.entry_amp_min.text())
            max_a = float(self.entry_amp_max.text())
            self.ax.set_ylim(min_a, max_a)
            self.update_background()
        except ValueError:
            QMessageBox.warning(self, "Parameter Error", "Please sequence numeric Y limits natively!")
            
    def auto_scale_amplitude(self):
        """v26.6: Automatically find local min/max for the VISIBLE window with 15% margin."""
        if self.audio_data is None: return
        try:
            # 1. Get current time window
            xlim = self.ax.get_xlim()
            
            # 2. Slice the current audio data to find peaks of what the user is actually seeing
            sr = self.sample_rate
            st_idx = int(max(0, xlim[0] * sr))
            en_idx = int(min(self.audio_data.shape[0], xlim[1] * sr))
            
            if en_idx > st_idx:
                # v26.6: Focus purely on the visible portion (Local Fit)
                visible_slice = self.audio_data[st_idx:en_idx]
                a_min = np.nanmin(visible_slice)
                a_max = np.nanmax(visible_slice)
                
                diff = a_max - a_min
                if diff == 0: diff = 1000.0 # prevent flat line stuck at zero
                
                # Professional 15% breathing room
                margin = diff * 0.15 
                
                self.entry_amp_min.setText(f"{a_min - margin:.1f}")
                self.entry_amp_max.setText(f"{a_max + margin:.1f}")
                self.apply_amp_limits()
                
        except Exception as e:
            print(f"Local Auto-Scale Error: {e}")

    def apply_freq_limits(self):
        # Allow frequency limits even if raw PCM is still buffering
        if self.sample_rate is None: return
        try:
            min_f = float(self.entry_freq_min.text())
            max_f = float(self.entry_freq_max.text())
            self.ax_spec.set_ylim(min_f, max_f)
            self.update_background()
        except ValueError:
            QMessageBox.warning(self, "Parameter Error", "Please sequence numeric spectrogram bounds natively!")

    def update_spec_appearance(self):
        """v28.1: Advanced Spectral Enhancements with Reactive Heatmaps & Reset."""
        if not hasattr(self, '_spec_img_obj') or self._spec_img_obj is None:
            return
            
        # Float factors mapped securely against central limits
        c      = self.slider_contrast.value() / 100.0
        b      = self.slider_brightness.value() / 100.0
        gamma  = self.slider_gamma.value() / 100.0
        sat    = self.slider_saturation.value() / 100.0
        invert = self.chk_invert.isChecked()
        cmap_name = self.combo_cmap.currentText()
        
        # ── Reactive Heatmap Logic (Multi-Channel vs Mono) ────────────────────
        is_rgb_view = getattr(self, '_spec_active_rgb', None) is not None
        
        if is_rgb_view and cmap_name == "Native Bearing":
            # 1. Bearing View (HSV DoA Mode) - Traditional AMMS visualizer
            rgb = (self._spec_active_rgb - 0.5) * c + 0.5 + b
            
            if sat != 1.0:
                gray = np.dot(np.clip(rgb[...,:3], 0, 1), [0.2989, 0.5870, 0.1140])[..., np.newaxis]
                rgb = np.clip(gray + sat * (rgb - gray), 0.0, 1.0)
            
            if gamma != 1.0 and gamma > 0.05:
                rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / gamma)
            
            if invert:
                rgb = 1.0 - rgb
            
            self._spec_img_obj.set_data(np.clip(rgb, 0.0, 1.0))
            
        elif is_rgb_view and cmap_name != "Native Bearing":
            # 2. INTENSITY View (Heatmap Mode) - Using the requested colormap
            # Even for multi-channel AMMS files, we can switch to a professional heatmap
            if getattr(self, '_spec_active_intensity', None) is not None:
                # Use stored intensity matrix
                intens = (self._spec_active_intensity - 0.5) * c + 0.5 + b
            else:
                # Fallback to luminance of the RGB view if intensity tracker missing
                intens = np.dot(np.clip(self._spec_active_rgb[...,:3], 0, 1), [0.2989, 0.5870, 0.1140])
                intens = (intens - 0.5) * c + 0.5 + b
            
            # Apply Gamma/Invert to intensity before colormapping for best contrast
            if gamma != 1.0 and gamma > 0.05:
                intens = np.power(np.clip(intens, 0.0, 1.0), 1.0 / gamma)
            if invert:
                intens = 1.0 - intens
            
            # Map Intensity -> RGB using the selected colormap
            try:
                mapper = plt.get_cmap(cmap_name)
                heatmap_rgb = mapper(np.clip(intens, 0, 1))
                self._spec_img_obj.set_data(heatmap_rgb)
            except:
                pass
                
        elif getattr(self, '_spec_active_mono', None) is not None:
            # 3. Mono fallback dB scale mapping
            center, span = -40.0, 80.0
            new_span = span / c if c > 0.01 else span
            shift = b * span
            new_vmin = (center - new_span / 2) - shift
            new_vmax = (center + new_span / 2) - shift
            self._spec_img_obj.set_cmap(cmap_name if cmap_name != "Native Bearing" else "magma")
            self._spec_img_obj.set_clim(vmin=new_vmin, vmax=new_vmax)
            
        self.canvas.draw_idle()
        self._bg_debounce.start(50)

    # ─────────────────────────────────────────────────────────
    #  Blit infrastructure
    # ─────────────────────────────────────────────────────────
    def on_resize(self, event):
        if self.audio_data is not None:
            self.update_background()

    def update_background(self):
        """Render everything static, then cache the raster for blitting."""
        was_vis  = (getattr(self, 'play_marker', None) is not None and
                    self.play_marker is not None and
                    self.play_marker.get_visible())

        for obj in [self.play_marker, self.play_text, self.spec_marker, getattr(self, 'play_text_sec', None)]:
            if obj is not None:
                obj.set_visible(False)

        self.canvas.draw()
        self.bg = self.canvas.copy_from_bbox(self.fig.bbox)

        if was_vis:
            for obj in [self.play_marker, self.play_text, self.spec_marker, getattr(self, 'play_text_sec', None)]:
                if obj is not None:
                    obj.set_visible(True)
            self.blit_playhead()
        
        # v10: Trigger debounced HD spectral refinement
        self.refine_timer.start(500) # Wait 500ms after last pan/zoom to refine

    def reset_spec_appearance(self):
        """v28.1: Restore all spectrogram sliders and filters to their initial defaults."""
        self.slider_contrast.blockSignals(True)
        self.slider_brightness.blockSignals(True)
        self.slider_gamma.blockSignals(True)
        self.slider_saturation.blockSignals(True)
        self.chk_invert.blockSignals(True)
        self.combo_cmap.blockSignals(True)
        
        self.slider_contrast.setValue(100)
        self.slider_brightness.setValue(0)
        self.slider_gamma.setValue(100)
        self.slider_saturation.setValue(100)
        self.chk_invert.setChecked(False)
        self.combo_cmap.setCurrentText("Native Bearing")
        
        self.slider_contrast.blockSignals(False)
        self.slider_brightness.blockSignals(False)
        self.slider_gamma.blockSignals(False)
        self.slider_saturation.blockSignals(False)
        self.chk_invert.blockSignals(False)
        self.combo_cmap.blockSignals(False)
        
        self.update_spec_appearance()

    def _on_hd_cols_changed(self, val):
        """v27.9: Save resolution to class-level so next windows remember it."""
        AudioAnnotatorApp.GLOBAL_HD_COLS = val

    def manual_refine_now(self):
        """User clicked the button to force an HD refinement regardless of Auto-Refine setting."""
        if self.audio_data is None: 
            return
        self.trigger_refinement(force=True, is_auto=False)

    def trigger_refinement(self, force=False, is_auto=False):
        """Starts background HD refinement or pulls from RAM cache if available."""
        if not hasattr(self, 'filepath'): return
        if self.sample_rate is None: return
        
        xlim = self.ax.get_xlim()
        vis_dur = xlim[1] - xlim[0]
        
        # v27.6: Define the target resolution from the UI
        target_cols = self.spin_hd_cols.value()
        
        # v17 Resource Logic: If auto-refine is off and not forced, just show a status and return
        if not self.chk_auto_refine.isChecked() and not force:
            # If it's already HD, keep the status, otherwise show 'Manual Mode'
            if not getattr(self, 'is_view_refined', False):
                self.lbl_refine_status.setText("🔳 View: Auto-Refine Off")
                self.lbl_refine_status.setStyleSheet("color: #9E9E9E; font-weight: bold;")
            return
        
        # v10 Logic: Don't refine if we already have a refined view for this window
        # v19: Bypassed if 'force' is True (Manual Refinement)
        if self.last_refined_xlim[0] is not None and not force:
            if np.allclose(xlim, self.last_refined_xlim, atol=1e-3):
                return
        
        # v12 Cache Logic: Check RAM for an existing refined slice with the CORRECT resolution
        # v27.3: The cache key now includes matching resolution (target_cols)
        cache_key = None
        for key in self.refined_cache.keys():
            # key is ((xmin, xmax), resolution)
            if np.allclose(xlim, key[0], atol=1e-3) and key[1] == target_cols:
                cache_key = key
                break
        
        if cache_key is not None and not force:
            # INSTANT SWAP: No thread needed!
            self._on_refinement_ready(self.refined_cache[cache_key], from_cache=True)
            return

        # v14 Master HD Bypass: If we have the full-file HD already, just extract the slice!
        # v19 Override: If manually 'forced', we bypass the cache to recalculate fresh Ultra-HD
        if self.is_master_hd and self.master_hd_data is not None and not force:
            # We don't need a thread because indexing a large matrix is instant
            rgb_full = self.master_hd_data["rgb"]
            t_full   = self.master_hd_data["t"]
            f_full   = self.master_hd_data["f"]
            
            if t_full is not None:
                # Find indices in the master matrix
                idx_start = np.searchsorted(t_full, xlim[0])
                idx_end   = np.searchsorted(t_full, xlim[1])
                
                # Safety check: if slice matches or is too small, fallback
                if idx_end > idx_start:
                    slice_payload = {
                        "rgb": rgb_full[:, idx_start:idx_end, :],
                        "t": t_full[idx_start:idx_end],
                        "f": f_full,
                        "is_mono": False
                    }
                    self._on_refinement_ready(slice_payload, from_cache=True)
                    return

        # Don't refine if we are viewing the whole file or a very large chunk
        max_dur = self.get_max_win_dur()
        if vis_dur > max_dur: return
        
        # v32.14: Adaptive RAM-aware resolution scaling
        _, ram_avail, _ = self.latest_resources
        
        if ram_avail < 1.0: # Red zone (< 1GB)
            target_cols = 2000
            self.lbl_refine_status.setText("🔳 View: SAFE MODE (2k)")
            self.lbl_refine_status.setStyleSheet("color: #E91E63; font-weight: bold;")
        elif ram_avail < 2.0: # Yellow zone (1-2GB)
            target_cols = 4000
            self.lbl_refine_status.setText("🔳 View: ECO MODE (4k)")
            self.lbl_refine_status.setStyleSheet("color: #FF9800; font-weight: bold;")
        else: # Green zone (> 2GB)
            # v32.15: Specific High-Density Warning for large 48kHz windows
            if self.sample_rate and self.sample_rate >= 44100 and vis_dur > 150.0:
                 self.lbl_refine_status.setText("🔳 View: Hi-Density (Slow)")
                 self.lbl_refine_status.setStyleSheet("color: #FF9800; font-weight: bold;")
            else:
                 self.lbl_refine_status.setText("🔳 View: Refining HD...")
                 self.lbl_refine_status.setStyleSheet("color: #2196F3; font-weight: bold;")
            
            # v27.7: Seamless Resolution Swap
            # If the view is already refined, instantly 'drop' back to low-res
            # so the user sees the new process starting immediately.
            if self.is_view_refined:
                self.is_view_refined = False
                self.last_refined_xlim = (None, None)
                self.update_background() # Re-draws the low-res NZP version base
        
        if self.current_refine_thread and self.current_refine_thread.isRunning():
            self.current_refine_thread.terminate()
            self.current_refine_thread.wait()
            
        # v24: Determine if we can use a direct RAM slice for refinement
        ram_data = None
        if self.audio_data is not None:
            # Extract the 300s window slice instantly in the main thread
            sr = self.sample_rate
            st_idx = int(xlim[0] * sr)
            en_idx = int(xlim[1] * sr)
            # Clip indices to available data
            st_idx = max(0, min(st_idx, self.audio_data.shape[0]))
            en_idx = max(st_idx, min(en_idx, self.audio_data.shape[0]))
            
            if en_idx > st_idx:
                ram_data = self.audio_data[st_idx:en_idx].copy()

        self.current_refine_thread = SpectralRefineThread(
            self.filepath, xlim[0], xlim[1], self.sample_rate, self.n_channels,
            audio_data=ram_data, target_cols=target_cols, is_auto=is_auto
        )
        self.current_refine_thread.finished.connect(self._on_refinement_ready)
        self.current_refine_thread.error.connect(self._on_refinement_error) # v23 stability
        self.current_refine_thread.start()

    def _on_refinement_error(self, err_msg):
        """v23: Handle refinement failures (e.g. network stall) by resetting the status."""
        self.lbl_refine_status.setText(f"⚠️ View Error")
        self.lbl_refine_status.setStyleSheet("color: #F44336; font-weight: bold;")
        print(f"Refinement Failure: {err_msg}")

    def _on_refinement_ready(self, payload, from_cache=False):
        """Seamlessly swap the blurry 'Overview' slice with sharp 'HD' data."""
        if not hasattr(self, '_spec_img_obj') or self._spec_img_obj is None:
            return
            
        xlim = self.ax.get_xlim() # current view bounds
        
        # Save to RAM cache for instant recall (limit to 10 entries)
        if not from_cache:
            self.last_refined_xlim = payload.get("xlim", xlim)
            self.last_refined_cols = self.spin_hd_cols.value() # v27.3: Track resolution
            
            # v27.3: Key now includes resolution (e.g. 6000 vs 12000)
            res_key = (tuple(xlim), self.last_refined_cols)
            self.refined_cache[res_key] = payload
            if len(self.refined_cache) > 10:
                # Remove oldest entry
                first_key = next(iter(self.refined_cache))
                del self.refined_cache[first_key]
            
        f = payload["f"]
        t = payload["t"]
        
        try:
            # If mono, handle clim update
            if payload["is_mono"]:
                data = payload["mono"]
                self._spec_active_mono = data # v16.1 tracker
                if self._spec_img_obj is not None:
                    self._spec_img_obj.set_data(data)
                    self._spec_img_obj.set_extent([t[0], t[-1], f[0], f[-1]])
                self.update_spec_appearance() # Triggers the clim update safely
            else:
                rgba = payload["rgb"]
                intensity = payload.get("intensity")
                self._spec_active_rgb = rgba 
                self._spec_active_intensity = intensity
                
                # v28.1.1: Ensure the image object boundaries are updated to the local refined window!
                if self._spec_img_obj is not None:
                    self._spec_img_obj.set_extent([t[0], t[-1], f[0], f[-1]])
                
                # Apply current appearance settings to the refined data
                self.update_spec_appearance()

            # v24.1: Instant Redraw & Status Update
            self.is_view_refined = True
            self.lbl_refine_status.setText("🔳 View: Refined (HD)")
            self.lbl_refine_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            # v28.2: Isolated Fast-Path Redraw
            # We ONLY call update_background() for manual requests (is_auto=False)
            # This makes Auto-Refine fluid because it skips the global re-render.
            is_auto = getattr(self.current_refine_thread, 'is_auto', False)
            if not is_auto:
                self.update_background()
            else:
                self.canvas.draw_idle()
            self.canvas.draw_idle()
            
        except (RuntimeError, AttributeError):
            # Object deleted or missing during background callback
            pass

    def _on_master_hd_progress(self, value, text):
        """Update the status label with the background upgrade progress."""
        self.is_upgrading_hd = True
        self.lbl_upgrade_status.setText(f"🚀 Master HD: {value}% - {text}")
        self.lbl_upgrade_status.setStyleSheet("color: #9C27B0; font-weight: bold;")

    def _on_master_hd_finished(self, filepath):
        """Successfully upgraded the .spec.npz to Master HD. Swap the UI base."""
        self.is_upgrading_hd = False
        self.is_master_hd = True
        self.lbl_upgrade_status.setText("🔊 Master HD: Ready (Cached)")
        self.lbl_upgrade_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.lbl_refine_status.setText("🔳 View: HD (Master)")
        self.lbl_refine_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        
        # We need to reload just the matrices from the new NPZ into our background Master HD storage
        try:
            with np.load(filepath.lower().replace(".wav", ".spec.npz"), allow_pickle=True) as arc:
                self.master_hd_data = {
                    "rgb": arc["rgb_base"],
                    "t": arc["t_spec"],
                    "f": arc["f_bins"]
                }
            
            # Immediately trigger a refinement check to swap current blurry view to HD
            self.trigger_refinement()
        except Exception as e:
            print(f"Failed to hot-swap Master HD: {e}")
            
        self.canvas.draw_idle()
        # Reset background after refinement so markers blit correctly
        self._bg_debounce.start(50) 
        
        # Mark final status
        self.lbl_pcm_status.setText("🔊 Audio: Ready")
        self.lbl_pcm_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.lbl_refine_status.setText("🔳 View: HD (Master)")
        self.lbl_refine_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
        # v13: Trigger 'eager' predictive pre-caching for neighbors while idle
        # Note: in this scope from_cache is not defined, we just start the timer
        self.predictive_timer.start(500) # Wait 0.5s of idleness (down from 2.0s)

    def blit_playhead(self):
        if not (hasattr(self, 'bg') and self.bg is not None):
            return
        self.canvas.restore_region(self.bg)
        if self.play_marker is not None and self.play_marker.get_visible():
            self.ax.draw_artist(self.play_marker)
            self.ax.draw_artist(self.play_text)
        if self.spec_marker is not None and self.spec_marker.get_visible():
            self.ax_spec.draw_artist(self.spec_marker)
            if getattr(self, 'play_text_sec', None) is not None:
                self.ax_spec.draw_artist(self.play_text_sec)
        self.canvas.blit(self.fig.bbox)
        self.canvas.flush_events()

    def start_predictive_caching(self):
        """Analyze current view and pre-calculate HD neighbors or parent windows."""
        if self.audio_data is None: return
        
        xlim = self.ax.get_xlim()
        dur  = xlim[1] - xlim[0]
        total_dur = self.get_total_duration()
        
        # Predict: User is most likely to 1) Zoom Out, 2) Move Forward, 3) Move Backward
        # target_blocks: list of (start, end)
        targets = []
        
        # A. Zoom-Out Prediction (Parent Window)
        # If we are zoomed in deep, pre-cache the adaptive parent window for instant zoom-out
        max_dur = self.get_max_win_dur()
        if dur < max_dur:
            center = (xlim[0] + xlim[1]) / 2
            p_start = max(0, center - max_dur / 2)
            p_end = min(total_dur, p_start + max_dur)
            if p_end - p_start < max_dur and p_end == total_dur:
                 p_start = max(0, p_end - max_dur)
            targets.append((p_start, p_end))

        # B. Neighbor Prediction (Move Forward/Backward)
        if xlim[1] + dur <= total_dur:
            targets.append((xlim[1], xlim[1] + dur))
        if xlim[0] - dur >= 0:
            targets.append((xlim[0] - dur, xlim[0]))
            
        # Filter out targets already in cache
        to_refine = []
        for t in targets:
            exists = any(np.allclose(t, k[0], atol=1e-3) for k in self.refined_cache.keys())
            if not exists:
                to_refine.append(t)
        
        if not to_refine: return
        
        # Start background refinement for the highest priority uncalculated target
        nxt_start, nxt_end = to_refine[0]
        
        if self.predictive_thread and self.predictive_thread.isRunning():
            return 
            
        self.predictive_thread = SpectralRefineThread(
            self.filepath, nxt_start, nxt_end, self.sample_rate, self.n_channels
        )
        self.predictive_thread.finished.connect(self._on_predictive_ready)
        self.predictive_thread.start()

    def _on_predictive_ready(self, payload):
        """Silently store predictive results in RAM for instant swaps later."""
        # Use a bounding box to find the key
        t = payload["t"]
        key = (t[0], t[-1])
        
        self.refined_cache[key] = payload
        if len(self.refined_cache) > 10:
            first_key = next(iter(self.refined_cache))
            del self.refined_cache[first_key]
        
        # If there's still idleness, the timer will re-trigger start_predictive_caching
        # (Actually, we should probably manually check for a second neighbor)
        self.predictive_timer.start(100) # Quick daisy-chain for second neighbor

    # ─────────────────────────────────────────────────────────
    #  Playback
    # ─────────────────────────────────────────────────────────
    def _start_playback_at(self, x):
        """Begin audio playback from position x (seconds)."""
        if self.audio_data is None:
            return
        start_idx = int(x * self.sample_rate)
        if not (0 <= start_idx < self.audio_data.shape[0]):
            return

        denoise_on = hasattr(self, 'check_denoise') and self.check_denoise.isChecked()

        if denoise_on:
            # Only process the CURRENT VIEW WINDOW to keep processing fast
            xlim = self.ax.get_xlim()
            end_s = min(xlim[1], self.audio_data.shape[0] / self.sample_rate)
            end_idx = int(end_s * self.sample_rate)
            raw_slice = (self.audio_data[start_idx:end_idx, 0] / 32768.0).astype(np.float32)

            if len(raw_slice) < 512:
                return

            strength = self.slider_denoise_intensity.value()

            # v38.0: Hide modal dialog when profile is LOCKED to allow fluid rapid-clicking
            show_progress = not (hasattr(self, 'check_lock_noise') and self.check_lock_noise.isChecked())
            prog = None
            if show_progress:
                prog = QProgressDialog("🎙️ Applying Noise Reduction...", None, 0, 0, self)
                prog.setWindowTitle("Noise Reduction")
                prog.setWindowModality(Qt.WindowModal)
                prog.setMinimumDuration(0)
                prog.setValue(0)
                prog.show()
                QApplication.processEvents()
            else:
                QApplication.setOverrideCursor(Qt.WaitCursor)
            
            try:
                # Run noise reduction
                cleaned = self._apply_noise_reduction(raw_slice, strength)
                
                # Chain Dynamic Boost (DRC)
                if hasattr(self, 'check_boost') and self.check_boost.isChecked():
                    boost_strength = self.slider_boost_intensity.value()
                    cleaned = self._apply_dynamic_boost(cleaned, boost_strength)
            finally:
                if prog:
                    prog.close()
                else:
                    QApplication.restoreOverrideCursor()

            self.playback_start_x   = x
            self.playback_start_sys = time.time()
            self._denoise_end_x     = end_s
            sd.play(cleaned, int(self.sample_rate * self.playback_speed))
        else:
            self.playback_start_x   = x
            self.playback_start_sys = time.time()
            self._denoise_end_x     = None
            
            play_data = (self.audio_data[start_idx:, 0] / 32768.0).astype(np.float32)

            # Apply Boost even if Noise Reduction is OFF
            if hasattr(self, 'check_boost') and self.check_boost.isChecked():
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    xlim = self.ax.get_xlim()
                    end_s = min(xlim[1], self.audio_data.shape[0] / self.sample_rate)
                    end_idx = int(end_s * self.sample_rate)
                    if end_idx > start_idx:
                        play_data = (self.audio_data[start_idx:end_idx, 0] / 32768.0).astype(np.float32)
                        boost_strength = self.slider_boost_intensity.value()
                        play_data = self._apply_dynamic_boost(play_data, boost_strength)
                        self._denoise_end_x = end_s
                finally:
                    QApplication.restoreOverrideCursor()

            sd.play(play_data, int(self.sample_rate * self.playback_speed))

        self.playback_timer.start(50)

    def _apply_noise_reduction(self, audio, strength):
        """
        Spectral noise reduction using the noisereduce library.
        - Uses the first 0.5s of the audio window as the noise sample.
        - prop_decrease (0.0-1.0) controls aggressiveness.
        - strength (1-10) maps to prop_decrease (0.1-1.0).
        """
        try:
            import noisereduce as nr

            sr = self.sample_rate

            # v37.0: Support Noise Locking
            if hasattr(self, 'check_lock_noise') and self.check_lock_noise.isChecked() and self._cached_noise_sample is not None:
                # Use the locked profile
                noise_sample = self._cached_noise_sample
            else:
                # Learn new noise from first 0.5s (default)
                noise_sample_len = min(int(sr * 0.5), len(audio) // 4)
                noise_sample = audio[:noise_sample_len]
                self._cached_noise_sample = noise_sample # Update cache for future locking

            # Map strength 1-10 → prop_decrease 0.1–1.0
            prop = strength / 10.0

            reduced = nr.reduce_noise(
                y=audio,
                sr=sr,
                y_noise=noise_sample,       # explicit noise reference
                prop_decrease=prop,         # how much noise to remove
                stationary=True,            # assume constant background hum
                n_fft=1024,                 # FFT window
            )

            # Normalize output
            max_val = np.max(np.abs(reduced))
            if max_val > 0.0001:
                reduced = (reduced / max_val * 0.9).astype(np.float32)

            return reduced

        except Exception:
            import traceback
            print(f"Noise Reduction Error: {traceback.format_exc()}")
            return audio

    def _update_denoise_line(self):
        """No-op: spectral noise reduction has no frequency-line overlay."""
        pass

    def _apply_dynamic_boost(self, audio, strength):
        """
        Apply Dynamic Range Compression (DRC) to boost quiet sounds.
        v37.0: RMS-based Gain Follower (No distortion).
        Adjusts overall volume envelope rather than individual wave peaks.
        """
        if len(audio) < 256:
            return audio
        try:
            # 1. Base Peak Normalization
            peak = np.max(np.abs(audio))
            if peak < 1e-6:
                return audio
            x = (audio / peak).astype(np.float32)
            
            # 2. Compute Smoothing Envelope (RMS) over ~50ms windows
            win_size = int(self.sample_rate * 0.05)
            # Moving average of squared signal
            sq = x**2
            rms = np.sqrt(np.convolve(sq, np.ones(win_size)/win_size, mode='same'))
            rms = np.maximum(rms, 1e-4) # Safety floor

            # 3. Dynamic Gain Calculation
            # p (0.0 to 0.75) controls how much we push the 'valley' up
            p = strength / 13.0
            gain = (1.0 / rms)**p
            
            # Cap gain to avoid blowing up silent static
            gain = np.minimum(gain, 15.0)

            # 4. Apply Gain
            boosted = x * gain
            
            # 5. Makeup Gain / Final Re-normalization
            final_peak = np.max(np.abs(boosted))
            if final_peak > 0:
                boosted = (boosted / final_peak * 0.9).astype(np.float32)
            
            return boosted
        except Exception as e:
            print(f"Smooth Boost Error: {e}")
            return audio


    def change_speed(self, text):
        old_speed = self.playback_speed
        self.playback_speed = float(text.replace("x", ""))
        if self.playback_timer.isActive():
            now = time.time()
            elapsed = (now - self.playback_start_sys) * old_speed
            # v31.0 Latency-corrected position
            current_x = self.playback_start_x + max(0, elapsed - self.latency_compensation)
            self.stop_audio()
            self._move_playhead_to(current_x)
            self._start_playback_at(current_x)

    def stop_audio(self):
        sd.stop()
        self.playback_timer.stop()
        self.paused = False
        for obj in [self.play_marker, self.play_text, self.spec_marker, getattr(self, 'play_text_sec', None)]:
            if obj is not None:
                obj.set_visible(False)
        self.blit_playhead()

    def toggle_pause(self):
        if self.audio_data is None:
            QMessageBox.information(self, "Buffering", "Audio is still buffering in the background. Please wait a few seconds...")
            return
        if self.playback_timer.isActive():
            self.playback_timer.stop()
            sd.stop()
            now = time.time()
            elapsed = (now - self.playback_start_sys) * self.playback_speed
            # v31.0: Record the ACOUSTIC position (what you just heard), not the system cursor.
            self.paused_x = self.playback_start_x + max(0, elapsed - self.latency_compensation)
            self.paused = True
            self.blit_playhead()
        elif self.paused:
            # v31.3: "Backtrack on Resume" - Rewind by 0.2s to provide auditory context
            # and compensate for hardware start-up latency.
            resume_x = max(0, self.paused_x - 0.25)
            self._move_playhead_to(resume_x)
            self._start_playback_at(resume_x)
            self.paused = False

    def _move_playhead_to(self, x):
        """Reposition the animated playhead markers to position x."""
        if self.play_marker is not None:
            verts = self.play_marker.get_xy()
            verts[0, 0] = verts[1, 0] = x
            if len(verts) > 4:
                verts[4, 0] = x
            verts[2, 0] = verts[3, 0] = x
            self.play_marker.set_xy(verts)
            self.play_marker.set_visible(True)
        if self.play_text is not None:
            ylim = self.ax.get_ylim()
            # v31.3: Dropped from 0.90 to 0.85 for extra 1080p headroom
            self.play_text.set_position((x, ylim[1] * 0.85))
            self.play_text.set_visible(True)
        if self.spec_marker is not None:
            self.spec_marker.set_xdata([x, x])
            self.spec_marker.set_visible(True)
        if getattr(self, 'play_text_sec', None) is not None:
            yspec = self.ax_spec.get_ylim()
            # Place the secondary marker exactly floating right above the bottom X-axis 
            self.play_text_sec.set_position((x, yspec[0] + (yspec[1] - yspec[0]) * 0.05))
            self.play_text_sec.set_visible(True)

    def update_playhead(self):
        if self.audio_data is None:
            return
        now = time.time()
        elapsed = (now - self.playback_start_sys) * self.playback_speed
        # v31.0: Latency Offset to sync visual playhead with acoustic output
        current_x = self.playback_start_x + max(0, elapsed - self.latency_compensation)
        duration  = self.audio_data.shape[0] / self.sample_rate

        if current_x >= duration:
            self.stop_audio()
            return
        
        # Stop at window end when Denoise is ON to prevent crash on next window
        denoise_end = getattr(self, '_denoise_end_x', None)
        if denoise_end is not None and current_x >= denoise_end:
            self.stop_audio()
            return

        # Auto-scroll camera if playhead hits right edge
        xlim = self.ax.get_xlim()
        if current_x > xlim[1]:
            width = xlim[1] - xlim[0]
            self.ax.set_xlim(current_x, current_x + width)
            
            try:
                self.entry_start_time.setText(f"{current_x:.3f}")
                self.entry_end_time.setText(f"{current_x + width:.3f}")
            except: pass
            
            self.update_background()
            xlim = self.ax.get_xlim()

        self._move_playhead_to(current_x)

        # Update text label
        margin = (xlim[1] - xlim[0]) * 0.08
        if current_x > xlim[1] - margin:
            self.play_text.set_horizontalalignment('right')
            if getattr(self, 'play_text_sec', None): self.play_text_sec.set_horizontalalignment('right')
        elif current_x < xlim[0] + margin:
            self.play_text.set_horizontalalignment('left')
            if getattr(self, 'play_text_sec', None): self.play_text_sec.set_horizontalalignment('left')
        else:
            self.play_text.set_horizontalalignment('center')
            if getattr(self, 'play_text_sec', None): self.play_text_sec.set_horizontalalignment('center')

        # v31.2: Relative Time Tracking
        # Top Label (Waveform): Relative to last click
        rel_x = current_x - self.listen_ref_x
        self.play_text.set_text(f"{rel_x:+.3f}s")
            
        # Bottom Label (Spectrogram): Absolute from start
        if getattr(self, 'play_text_sec', None) is not None:
            self.play_text_sec.set_text(f"{current_x:.3f}s")

        self.blit_playhead()

    def update_latency_from_ui(self, val):
        """v31.1: Live sync for the manual latency calibration tool."""
        self.latency_compensation = val

    # ─────────────────────────────────────────────────────────
    #  Mouse events
    # ─────────────────────────────────────────────────────────
    def on_click(self, event):
        if event.inaxes not in (self.ax, self.ax_spec):
            return
        modifiers = QApplication.keyboardModifiers()
        if not bool(modifiers & Qt.ControlModifier):
            return
        if event.button == 3:
            self.toggle_pause()
            return
        if event.button == 1 and event.xdata is not None:
            if self.audio_data is None:
                QMessageBox.information(self, "Buffering", "Audio is still buffering in the background. Please wait a few seconds...")
                return
            x = event.xdata
            self.stop_audio()
            
            # v31.2: Update relative listening reference
            self.listen_ref_x = x
            # Since secondary_xaxis functions are lambdas referencing self.listen_ref_x,
            # we just need to force a redraw or refresh the axis object.
            # Matplotlib secondary axes can be tricky; the most robust way is to re-set the functions
            # but usually a simple relim/redraw works if they are captured in scope.
            # Let's re-set for absolute safety.
            try:
                self.ax_top.remove()
                self.ax_top = self.ax.secondary_xaxis('top', functions=(
                    lambda val: val - self.listen_ref_x,
                    lambda val: val + self.listen_ref_x
                ))
                self.ax_top.set_xlabel("Relative Time (s) [from listen click]")
                self.ax_top.xaxis.label.set_color('#1976D2')
                self.ax_top.tick_params(axis='x', colors='#1976D2')
            except: pass

            self._move_playhead_to(x)
            self.blit_playhead()
            self._start_playback_at(x)

    def on_scroll(self, event):
        if event.inaxes not in (self.ax, self.ax_spec):
            return
        if event.xdata is None:
            return
        modifiers = QApplication.keyboardModifiers()
        if not bool(modifiers & Qt.ControlModifier):
            return
        if self.get_total_duration() <= 0:
            return

        scale  = 1.3
        xlim   = self.ax.get_xlim()
        xdata  = event.xdata
        rng    = xlim[1] - xlim[0]
        lfrac  = (xdata - xlim[0]) / rng

        if event.button == 'up':
            new_rng = rng / scale
        elif event.button == 'down':
            new_rng = rng * scale
        else:
            return

        new_xmin = xdata - lfrac * new_rng
        new_xmax = xdata + (1 - lfrac) * new_rng

        # v10 Safety: Prevent zooming out beyond adaptive limit
        visible_dur = new_xmax - new_xmin
        max_dur = self.get_max_win_dur()
        if visible_dur > max_dur:
            # Re-scale to exactly max_dur centered on mouse
            new_xmin = xdata - lfrac * max_dur
            new_xmax = xdata + (1 - lfrac) * max_dur
        
        dur = self.get_total_duration()

        if new_rng >= dur:
            new_xmin, new_xmax = 0, dur
        else:
            if new_xmin < 0:
                new_xmin, new_xmax = 0, new_rng
            if new_xmax > dur:
                new_xmax = dur
                new_xmin = dur - new_rng

        self.ax.set_xlim(new_xmin, new_xmax)   # ax_spec follows via sharex
        
        # Sync the dock widget dynamically
        try:
            self.entry_start_time.setText(f"{new_xmin:.3f}")
            self.entry_duration.setText(f"{new_xmax - new_xmin:.3f}")
        except: pass
        
        # Non-blocking: let Qt batch all scroll ticks into one render frame.
        # The blit background is refreshed 200 ms after scrolling stops.
        self.canvas.draw_idle()
        self._bg_debounce.start(200)

    # ─────────────────────────────────────────────────────────
    #  Annotation
    # ─────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────
    #  Annotation IO Auto-Save & Storage Native Persistence Modules
    # ─────────────────────────────────────────────────────────
    def _save_annotations(self):
        """Silently flush all active physical annotations to the bounding CSV logic file."""
        if not hasattr(self, 'annotation_file') or not self.annotation_file: return
        try:
            with open(self.annotation_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['label', 'xmin', 'xmax', 'hex_color', 'weight'])
                for ann in self.annotations:
                    assigned_color = self.class_colors.get(ann['label'].strip().lower(), '#FFFFFF')
                    hex_val = mcolors.to_hex(assigned_color)
                    writer.writerow([ann['label'], ann['xmin'], ann['xmax'],
                                     hex_val, ann.get('weight', 0.0)])
        except Exception as e:
            print(f"Failed background auto-save: {e}")

    def _load_annotations(self):
        """Silently probe the directory for matching historical tracking markers."""
        if not hasattr(self, 'annotation_file') or not self.annotation_file: return
        if not os.path.exists(self.annotation_file): return
        try:
            with open(self.annotation_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        lbl  = row[0]
                        xmin = float(row[1])
                        xmax = float(row[2])
                        force_color = row[3] if len(row) > 3 else None
                        weight = float(row[4]) if len(row) > 4 else 0.0
                        self._add_annotation_to_tree(xmin, xmax, lbl,
                            force_color=force_color, weight=weight, autosave=False)
            self.update_background()
        except Exception as e:
            QMessageBox.warning(self, "Auto-Load Error",
                f"Found previous annotations but couldn't load them:\n{e}")

    def on_select_interval(self, xmin, xmax):
        if (xmax - xmin) < 0.05:
            return
        base_dt = self.get_base_datetime()
        if not base_dt:
            return

        dlg = AnnotationEditDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return

        self._add_annotation_to_tree(xmin, xmax, dlg.class_label,
                                     weight=dlg.weight, autosave=True)

    def _add_annotation_to_tree(self, xmin, xmax, class_label, force_color=None,
                                weight=0.0, autosave=True):
        base_dt = self.get_base_datetime()
        if not base_dt: return

        # ── Time Sync Correction ────────────────────────────────
        unsync_start_dt = base_dt + timedelta(seconds=xmin)
        unsync_end_dt   = base_dt + timedelta(seconds=xmax)
        
        sync_active = False
        log_info = ""
        
        if self.amms_timing:
            dt_s = self.amms_timing.sample2time(xmin * self.sample_rate)
            dt_e = self.amms_timing.sample2time(xmax * self.sample_rate)
            if dt_s and dt_e:
                # Apply 1-hour Bavaria offset if the base_dt implies it?
                # Actually user said "apply the time stap which is corrected by time sync"
                # If amms_timing.sample2time returns absolute UTC, we should align it.
                # Usually log ptime is UTC. If base_dt is local, we adjust.
                
                # Check the difference to decide if we need to apply the same offset as base_dt
                # This depends on how get_base_datetime is implemented.
                # For now, we assume amms_timing handles the core sync logic (sample mapping).
                start_dt = dt_s
                end_dt = dt_e
                sync_active = True
                
                # Extract Log Metadata for summary
                meta = self.amms_timing.get_metadata_for_sample(xmin * self.sample_rate)
                if meta:
                    # v32.2: Log-relative sample indices (Absolute hardware counters)
                    log_start_sample = int(xmin * self.sample_rate + self.amms_timing.audio_start_sample)
                    log_end_sample   = int(xmax * self.sample_rate + self.amms_timing.audio_start_sample)
                    
                    # v32.2: Decrypt Quality (Microflown/AMMS Standard)
                    q_hex = meta.get('quality', '00000000')
                    try:
                        q_val = int(q_hex, 16)
                        # Check Bit 16: 0x00010000 usually indicates PTP/GPS Lock
                        if (q_val & 0x00010000) or q_hex.startswith("0001"):
                            q_desc = "Locked (GPS/PTP)"
                        else:
                            q_desc = "Free Running"
                    except:
                        q_desc = "Unknown"

                    # v32.2: Identify if the point is a direct lock or an estimated anchor
                    is_abs = meta.get('_source', '') in ['ptime', 'tsyncTime']
                    # Any point that was shifted by the 32.2 engine will have a high value but might have a 'time' source
                    # We can use the quality bit as a hint too.
                    
                    source = meta.get('_source', 'N/A') if meta else 'N/A'
                    if self.amms_timing.anchor_source and self.amms_timing.anchor_source != source:
                        source_display = f"{source} (Anchored by {self.amms_timing.anchor_source})"
                    else:
                        source_display = source

                    log_info = "<br><b>Log Synchronization:</b><br>"
                    log_info += f"• Time Source: {source_display}<br>"
                    log_info += f"• Sync Status: {q_desc}<br>"
                    
                    # Magnitude check matching amms_timing.py
                    t_val_micros = (start_dt - datetime(1970, 1, 1)).total_seconds() * 1e6
                    if t_val_micros > 1.0e13: # Absolute timeline active
                        status_str = "High-Precision Absolute" if is_abs else "Back-Filled Absolute"
                        log_info += f"• Timeline: <font color='green'>{status_str}</font><br>"
                        
                        # Calculate exact effective hardware frequency
                        dt_synced_dur = (dt_e - dt_s).total_seconds()
                        if dt_synced_dur > 0:
                            samp_delta = (xmax - xmin) * self.sample_rate
                            effective_fs = samp_delta / dt_synced_dur
                            log_info += f"• Measured Rate: {effective_fs:.4f} Hz<br>"
                    else:
                        log_info += "• Timeline: Relative Uptime (1970 Epoch)<br>"
                        
                    log_info += f"• Hardware fs: {int(self.amms_timing.log_fs)} Hz<br>"
                    log_info += f"• Start Sample: {log_start_sample}<br>"
                    log_info += f"• End Sample: {log_end_sample}<br>"
            else:
                start_dt = unsync_start_dt
                end_dt = unsync_end_dt
        else:
            start_dt = unsync_start_dt
            end_dt = unsync_end_dt

        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        end_str   = end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")

        # ── Sync Feedback Message Box ───────────────────────────
        if autosave: # Only show during active creation, skip during bulk load
            msg = f"<b>Class:</b> {class_label}<br>"
            if sync_active:
                msg += "<font color='green'>✅ <b>Full Interval Time-Sync Applied</b></font><br><br>"
                msg += f"<b>Start (Unsynced):</b> {unsync_start_dt.strftime('%H:%M:%S.%f')}<br>"
                msg += f"<b>Start (Synced):</b> {start_dt.strftime('%H:%M:%S.%f')}<br>"
                
                msg += f"<br><b>End (Unsynced):</b> {unsync_end_dt.strftime('%H:%M:%S.%f')}<br>"
                msg += f"<b>End (Synced):</b> {end_dt.strftime('%H:%M:%S.%f')}<br>"
                
                start_diff = (start_dt - unsync_start_dt).total_seconds()
                end_diff   = (end_dt - unsync_end_dt).total_seconds()
                
                real_dur = (end_dt - start_dt).total_seconds()
                nom_dur  = (xmax - xmin)
                
                msg += f"<br><b>Drift at Start:</b> {start_diff:+.6f} s"
                msg += f"<br><b>Drift at End:</b> {end_diff:+.6f} s"
                msg += f"<br><b>Corrected Interval Duration:</b> {real_dur:.6f} s<br>"
                msg += f"(Nominal: {nom_dur:.6f} s)"
                msg += log_info
            else:
                msg += "<font color='orange'>⚠️ No local LOG file found. Using default timing.</font>"
            
            QMessageBox.information(self, "Label Synchronization Summary", msg)

        if force_color:
            self.class_colors[class_label.strip().lower()] = force_color
            color = force_color
        else:
            color = self.get_color_for_class(class_label)

        row_num  = self.tree.topLevelItemCount() + 1
        disp_txt = f"{class_label}\n[{row_num}]"

        face_rgba = mcolors.to_rgba(color, alpha=0.3)
        edge_rgba = mcolors.to_rgba(color, alpha=1.0)

        # Span on waveform
        span = self.ax.axvspan(xmin, xmax,
                               facecolor=face_rgba, edgecolor=edge_rgba,
                               linestyle='--', linewidth=2.5)
        txt = self.ax.text(
            (xmin + xmax) / 2, self.ax.get_ylim()[1] * 0.70, 
            disp_txt, horizontalalignment='center',
            color='black', weight='bold', fontsize=10, clip_on=False,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

        # Matching span on spectrogram
        spec_span = self.ax_spec.axvspan(xmin, xmax,
                                         facecolor=face_rgba, edgecolor=edge_rgba,
                                         linestyle='--', linewidth=2.5)

        self.spans.append((span, txt, spec_span))

        hex_color = mcolors.to_hex(color)
        
        # v32.4: Calculate absolute log-relative sample numbers
        if self.amms_timing:
            start_sample = int(xmin * self.sample_rate + self.amms_timing.audio_start_sample)
            end_sample   = int(xmax * self.sample_rate + self.amms_timing.audio_start_sample)
            start_sample_str = str(start_sample)
            end_sample_str   = str(end_sample)
        else:
            start_sample_str = "N/A"
            end_sample_str   = "N/A"

        # New Column Order: [Row, S_Samp, E_Samp, S_DT, E_DT, Label, Color, Weight]
        item = QTreeWidgetItem([
            str(row_num), 
            start_sample_str, 
            end_sample_str, 
            start_str, 
            end_str, 
            class_label, 
            "", 
            f"{float(weight):.2f}"
        ])
        
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(hex_color))
        item.setIcon(6, QIcon(pixmap))
        
        # v32.11: Center-align ALL columns for total aesthetic symmetry
        for col in range(8):
            item.setTextAlignment(col, Qt.AlignCenter)
            
        self.tree.addTopLevelItem(item)

        self.annotations.append({
            "id": id(item), "start": start_str, "end": end_str,
            "label": class_label, "xmin": xmin, "xmax": xmax,
            "weight": float(weight), "item": item
        })

        if autosave:
            self.update_background()
            self._save_annotations()
        
        # v35.3: Sync AI button states
        self._refresh_aibot_ui_state()

    def show_context_menu(self, position):
        selected = self.tree.selectedItems()
        if not selected:
            return
        
        menu = QMenu()
        
        # Only allow 'Edit' if exactly one item is selected
        if len(selected) == 1:
            item = selected[0]
            is_hidden = item.data(0, Qt.UserRole) == True
            edit_action   = menu.addAction("✏️  Edit Annotation")
            toggle_action = menu.addAction("👁️  Show Interval" if is_hidden else "👁️  Hide Interval")
            menu.addSeparator()
            delete_action = menu.addAction("🗑️  Delete Row")
            
            action = menu.exec_(self.tree.viewport().mapToGlobal(position))
            if action == edit_action:
                self.edit_annotation(item)
            elif action == toggle_action:
                self._toggle_interval_visibility(item)
            elif action == delete_action:
                self.delete_annotation()
        else:
            # v36.7: Bulk Visibility and Multi-delete
            hide_action   = menu.addAction(f"👁️  Hide {len(selected)} Selected Intervals")
            show_action   = menu.addAction(f"👁️  Show {len(selected)} Selected Intervals")
            menu.addSeparator()
            delete_action = menu.addAction(f"🗑️  Delete {len(selected)} Selected Rows")
            
            action = menu.exec_(self.tree.viewport().mapToGlobal(position))
            if action == hide_action:
                self._set_intervals_visibility(selected, make_hidden=True)
            elif action == show_action:
                self._set_intervals_visibility(selected, make_hidden=False)
            elif action == delete_action:
                self.delete_annotation()

    def _toggle_interval_visibility(self, item):
        self._set_intervals_visibility([item])

    def _set_intervals_visibility(self, items, make_hidden=None):
        """v36.9: Batch visibility toggle for multiple intervals."""
        for item in items:
            idx = next((i for i, a in enumerate(self.annotations) if a["item"] == item), None)
            if idx is None: continue
            
            # If make_hidden is None, toggle individually (old behavior logic)
            current_state = item.data(0, Qt.UserRole) == True
            new_state = make_hidden if make_hidden is not None else (not current_state)
            
            item.setData(0, Qt.UserRole, new_state)
            
            # 1. Update Plot Artists
            if idx < len(self.spans):
                artists = self.spans[idx]
                for art in artists:
                    if art is not None:
                        art.set_visible(not new_state)
            
            # 2. Update Row Styling
            color = QColor("gray") if new_state else QColor("black")
            for i in range(self.tree.columnCount()):
                item.setForeground(i, QBrush(color))
                font = item.font(i)
                font.setItalic(new_state)
                item.setFont(i, font)
        
        self.canvas.draw_idle()

    def on_item_double_clicked(self, item, col):
        """Focus the view on the double-clicked annotation interval."""
        if self.audio_data is None:
            return
        idx = next((i for i, a in enumerate(self.annotations) if a["item"] == item), None)
        if idx is None:
            return
        
        ann = self.annotations[idx]
        xmin, xmax = ann["xmin"], ann["xmax"]
        center = (xmin + xmax) / 2
        
        # Determine the target window duration (use setting if possible)
        max_dur = self.get_max_win_dur()
        try:
            win_dur = float(self.entry_duration.text())
        except:
            win_dur = max_dur
            
        total_dur = self.audio_data.shape[0] / self.sample_rate
        
        # If the annotation itself is longer than the window, fit it exactly
        # otherwise center the window around it.
        # v10 Safety: Always respect the hardware-aware duration limit
        target_span = min(max_dur, max(win_dur, (xmax - xmin) * 1.1))
        
        new_xmin = center - target_span / 2
        new_xmax = center + target_span / 2
        
        # Clamp to audio bounds
        if new_xmin < 0:
            new_xmin = 0
            new_xmax = min(target_span, total_dur)
        if new_xmax > total_dur:
            new_xmax = total_dur
            new_xmin = max(0, total_dur - target_span)
            
        self.ax.set_xlim(new_xmin, new_xmax)
        self.entry_start_time.setText(f"{new_xmin:.3f}")
        self.entry_end_time.setText(f"{new_xmax:.3f}")
        self.entry_duration.setText(f"{new_xmax - new_xmin:.3f}")
        
        # Trigger the highlight fade animation
        self.active_fades[idx] = 1.0  # Start fully opaque
        if not self.fade_timer.isActive():
            self.fade_timer.start(50) # 20 fps animation

        self.update_background()

    def process_fades(self):
        """Timer callback to incrementally reduce alpha of highlighted spans."""
        if not self.active_fades:
            self.fade_timer.stop()
            return
            
        to_remove = []
        for idx, alpha in self.active_fades.items():
            new_alpha = alpha - 0.1  # reduce by 10% per tick
            if new_alpha <= 0.3:
                new_alpha = 0.3
                to_remove.append(idx)
            
            self.active_fades[idx] = new_alpha
            
            # Apply to both waveform and spectrogram spans
            if idx < len(self.spans):
                span, txt, spec_span = self.spans[idx]
                curr_color = span.get_facecolor() # returns RGBA
                # Update alpha in the RGBA tuple natively
                new_face = (curr_color[0], curr_color[1], curr_color[2], new_alpha)
                span.set_facecolor(new_face)
                if spec_span is not None:
                    spec_span.set_facecolor(new_face)
                
        for idx in to_remove:
            del self.active_fades[idx]
            
        self.update_background()

    def edit_annotation(self, item):
        idx = next((i for i, a in enumerate(self.annotations) if a["item"] == item), None)
        if idx is None:
            return
        ann = self.annotations[idx]

        hw_offset = self.amms_timing.audio_start_sample if self.amms_timing else 0
        curr_hw_start = ann["xmin"] * self.sample_rate + hw_offset
        curr_hw_end   = ann["xmax"] * self.sample_rate + hw_offset

        dlg = AnnotationEditDialog(
            self,
            class_label=ann["label"],
            weight=ann.get("weight", 0.0),
            start_s=ann["xmin"],
            end_s=ann["xmax"],
            start_samp=curr_hw_start,
            end_samp=curr_hw_end,
            sample_rate=self.sample_rate,
            hardware_offset=hw_offset
        )
        if dlg.exec_() != QDialog.Accepted:
            return

        new_label = dlg.class_label
        new_weight = dlg.weight
        new_xmin = dlg.start_s
        new_xmax = dlg.end_s

        if new_xmin < 0 or new_xmax <= new_xmin:
            QMessageBox.warning(self, "Invalid Range",
                "Resulting time span is invalid.")
            return

        # ── Time Sync Re-Alignment ───────────────────
        # Since the user edited 'relative seconds', we find the absolute times
        # from the log for these specific points.
        base_dt = self.get_base_datetime()
        sync_active = False
        if self.amms_timing:
            dt_s = self.amms_timing.sample2time(new_xmin * self.sample_rate)
            dt_e = self.amms_timing.sample2time(new_xmax * self.sample_rate)
            if dt_s and dt_e:
                new_start = dt_s.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                new_end   = dt_e.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                sync_active = True
            else:
                new_start = (base_dt + timedelta(seconds=new_xmin)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                new_end   = (base_dt + timedelta(seconds=new_xmax)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        else:
            new_start = (base_dt + timedelta(seconds=new_xmin)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            new_end   = (base_dt + timedelta(seconds=new_xmax)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # ── Sync Feedback Summary for Edits ────────────────────
        msg = f"<b>Class:</b> {new_label}<br>"
        if sync_active:
            msg += "<font color='green'>✅ <b>Seconds Synced to Hardware Clock</b></font><br><br>"
            msg += f"<b>New Start Time:</b> {new_start}<br>"
            msg += f"<b>New End Time:</b> {new_end}<br>"
            
            # v32.3: Sample Number Comparison (Log Context)
            old_start_s = int(ann["xmin"] * self.sample_rate + self.amms_timing.audio_start_sample)
            new_start_s = int(new_xmin * self.sample_rate + self.amms_timing.audio_start_sample)
            old_end_s   = int(ann["xmax"] * self.sample_rate + self.amms_timing.audio_start_sample)
            new_end_s   = int(new_xmax * self.sample_rate + self.amms_timing.audio_start_sample)
            
            msg += "<br><b>Hardware Sample Comparison (Log):</b><br>"
            msg += f"• Start: {old_start_s} → <font color='blue'>{new_start_s}</font><br>"
            msg += f"• End: {old_end_s} → <font color='blue'>{new_end_s}</font><br>"
            
            # Show drift comparison (Synced vs Naive)
            naive_start = base_dt + timedelta(seconds=new_xmin)
            naive_end   = base_dt + timedelta(seconds=new_xmax)
            
            diff_s = (dt_s - naive_start).total_seconds()
            diff_e = (dt_e - naive_end).total_seconds()
            
            msg += f"<br><b>Start Drift Correction:</b> {diff_s:+.4f} s"
            msg += f"<br><b>End Drift Correction:</b> {diff_e:+.4f} s"
        else:
            msg += "<font color='orange'>⚠️ No log found. Times calculated naively from file start.</font>"
        
        QMessageBox.information(self, "Edit Synchronization Summary", msg)

        # ── Apply changes to internal dict ───────────────────────
        ann["label"]  = new_label
        ann["weight"] = new_weight
        ann["start"]  = new_start
        ann["end"]    = new_end
        ann["xmin"]   = new_xmin
        ann["xmax"]   = new_xmax

        # ── Update tree columns ──────────────────────────────────
        row_num = self.tree.indexOfTopLevelItem(item) + 1
        
        # Calculate Log Samples for display
        if self.amms_timing:
            new_start_samp = int(new_xmin * self.sample_rate + self.amms_timing.audio_start_sample)
            new_end_samp   = int(new_xmax * self.sample_rate + self.amms_timing.audio_start_sample)
            ss_str, es_str = str(new_start_samp), str(new_end_samp)
        else:
            ss_str, es_str = "N/A", "N/A"

        item.setText(0, str(row_num))
        item.setText(1, ss_str)
        item.setText(2, es_str)
        item.setText(3, new_start)
        item.setText(4, new_end)
        item.setText(5, new_label)
        item.setText(7, f"{float(new_weight):.2f}")
        
        # v32.11: Maintain Center-alignment across all columns
        for col in range(8):
            item.setTextAlignment(col, Qt.AlignCenter)
        
        # Update icon if color changed (in case label changed)
        color = self.get_color_for_class(new_label)
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(mcolors.to_hex(color)))
        item.setIcon(6, QIcon(pixmap))

        # ── Reposition graph spans ───────────────────────────────
        new_color = self.get_color_for_class(new_label)
        span, txt, spec_span = self.spans[idx]
        face_rgba = mcolors.to_rgba(new_color, alpha=0.3)
        edge_rgba = mcolors.to_rgba(new_color, alpha=1.0)

        # Move waveform span
        span.set_xy([[new_xmin, 0], [new_xmin, 1],
                     [new_xmax, 1], [new_xmax, 0], [new_xmin, 0]])
        span.set_facecolor(face_rgba)
        span.set_edgecolor(edge_rgba)
        spec_span.set_xy([[new_xmin, 0], [new_xmin, 1],
                          [new_xmax, 1], [new_xmax, 0], [new_xmin, 0]])
        spec_span.set_facecolor(face_rgba)
        spec_span.set_edgecolor(edge_rgba)
        txt.set_position(((new_xmin + new_xmax) / 2, self.ax.get_ylim()[1] * 0.82))
        txt.set_text(f"{new_label}\n[{row_num}]")

        hex_color = mcolors.to_hex(new_color)
        px = QPixmap(16, 16)
        px.fill(QColor(hex_color))
        item.setIcon(6, QIcon(px))

        self.update_background()
        self._save_annotations()

    def delete_annotation(self, specific_item=None):
        """v36.3: Robust Bulk Deletion with reverse-index processing."""
        items_to_del = [specific_item] if specific_item else self.tree.selectedItems()
        if not items_to_del:
            return

        # 1. Map items to their current indices in self.annotations
        indices = []
        for item in items_to_del:
            idx = next((i for i, a in enumerate(self.annotations) if a["item"] == item), None)
            if idx is not None:
                indices.append(idx)
        
        if not indices:
            return
            
        # 2. Sort indices in REVERSE order to prevent shifting bugs
        indices.sort(reverse=True)
        
        # 3. Batch Remove from Data & Matplotlib
        for idx in indices:
            # Clean up plot artists
            if idx < len(self.spans):
                artists = self.spans[idx]
                for artist in artists:
                    if artist is not None:
                        try:
                            artist.remove()
                        except:
                            pass
                del self.spans[idx]
            
            # Clean up data list
            item_to_take = self.annotations[idx]["item"]
            row_to_take = self.tree.indexOfTopLevelItem(item_to_take)
            if row_to_take != -1:
                self.tree.takeTopLevelItem(row_to_take)
            
            del self.annotations[idx]

        # 4. Batch Re-numbering (Once at the end)
        for i, ann in enumerate(self.annotations):
            row_idx = i + 1 
            # Update Plot Label
            _, txt, _ = self.spans[i]
            txt.set_text(f'{ann["label"]}\n[{row_idx}]')
            # Update Tree Column 0
            if ann['item']:
                ann['item'].setText(0, str(row_idx))
                ann['item'].setTextAlignment(0, Qt.AlignCenter)
        self.update_background()
        self._save_annotations()
        self._refresh_aibot_ui_state()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_annotation()
        else:
            super().keyPressEvent(event)

    # ─────────────────────────────────────────────────────────
    #  Export
    # ─────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────
    #  AI Bot Handlers (v30: Silent Expert Operator)
    # ─────────────────────────────────────────────────────────
    def _on_aibot_toggled(self, checked):
        """Enable or disable the AI Bot interface."""
        if checked:
            self.btn_enable_aibot.setText("🤖 AI Bot Enabled")
            self.btn_enable_aibot.setStyleSheet("font-weight: bold; padding: 5px; color: #FFFFFF; background-color: #673AB7;")
        else:
            self._clear_ghost_spans()
            self.btn_enable_aibot.setText("🤖 Enable AI Bot")
            self.btn_enable_aibot.setStyleSheet("font-weight: bold; padding: 5px; color: #673AB7;")
            
            if self.current_aibot_thread and self.current_aibot_thread.isRunning():
                self.current_aibot_thread.terminate()
        
        self._refresh_aibot_ui_state()

    def _refresh_aibot_ui_state(self):
        """Unified UI state refresh for AI buttons."""
        is_checked = self.btn_enable_aibot.isChecked()
        has_min_labels = len(self.annotations) >= 2
        
        self.btn_yamnet.setEnabled(is_checked)
        self.btn_train_expert.setEnabled(is_checked and has_min_labels)
        self.btn_view_results.setEnabled(is_checked and bool(self.discovered_intervals))


    def _on_aibot_yamnet_clicked(self):
        """Phase 1: YAMNet Full-Dataset Auto-Label."""
        if self.audio_data is None:
            QMessageBox.warning(self, "AI Error", "Audio data must be loaded.")
            return

        xlim = self.ax.get_xlim()
        start_s = max(0, xlim[0])
        end_s = min(self.audio_data.shape[0]/self.sample_rate, xlim[1])
        
        self.discovered_intervals = []
        
        # v36.6: Restore the Premium Dialog for YAMNet
        self.ai_progress_dialog = AIProgressDialog("AI Bot - Orchestrating Discovery...", "Loading YAMNet model...", self)
        self.ai_progress_dialog.show()
        
        self.current_aibot_thread = AIBotInferenceThread(
            'yamnet_discovery', self.filepath, self.sample_rate, self.n_channels,
            self.audio_data, None, {}, (start_s, end_s)
        )
        self.current_aibot_thread.progress.connect(self._on_aibot_progress)
        self.current_aibot_thread.log_signal.connect(self._on_aibot_log)
        self.current_aibot_thread.discovery_event.connect(self._on_aibot_event_found)
        self.current_aibot_thread.finished.connect(self._on_aibot_finished)
        self.current_aibot_thread.error.connect(self._on_aibot_error)
        self.current_aibot_thread.start()

    def _on_aibot_progress(self, val, text):
        if not self.ai_progress_dialog: return
        self.ai_progress_dialog.setValue(val)
        self.ai_progress_dialog.setLabelText(text)

    def _on_aibot_log(self, text):
        if not self.ai_progress_dialog: return
        self.ai_progress_dialog.log_view.append(text)
        self.ai_progress_dialog.log_view.verticalScrollBar().setValue(
            self.ai_progress_dialog.log_view.verticalScrollBar().maximum()
        )

    def _on_aibot_view_results_clicked(self):
        """Show the Discovery results in a popup."""
        if not self.discovered_intervals:
             QMessageBox.information(self, "AI Bot", "No discovery results available yet.")
             return
        dlg = AIDiscoveryDialog(self.discovered_intervals, self)
        
        def on_approve():
             selected = dlg.get_selected_intervals()
             dlg.accept()
             if not selected: return
             
             progress = QProgressDialog("Applying labels...", "Cancel", 0, len(selected), self)
             progress.setWindowTitle("Auto Approve")
             progress.setWindowModality(Qt.WindowModal)
             progress.setMinimumDuration(0)
             
             count = 0
             for start, end, desc in selected:
                 if progress.wasCanceled():
                     break
                 
                 label_text = desc.split("|")[0] if "|" in desc else desc
                 weight = 5.0
                 if "|" in desc:
                     try: weight = float(desc.split("|")[1]) * 10
                     except: pass
                 
                 # Draw visually bypassing standard auto-draw logic if it is slow, 
                 # but _add_annotation_to_tree takes care of it nicely
                 self._add_annotation_to_tree(start, end, label_text, weight=weight, autosave=False)
                 count += 1
                 progress.setValue(count)
                 QApplication.processEvents()
                 
             # Commit massive save state after loop completes
             self._save_annotations()
             QMessageBox.information(self, "Auto Approve", f"Successfully promoted {count} events to annotations!")
             
        dlg.btn_auto_approve.clicked.connect(on_approve)
        dlg.exec_()
        # Always clear purple spans when dialog closes (approved or cancelled)
        self._clear_ghost_spans()

    def _clear_ghost_spans(self):
        """Remove all temporary purple AI discovery highlights from the waveform."""
        for gs in getattr(self, 'ghost_spans', []):
            try:
                gs[0].remove()   # axvspan
            except Exception:
                pass
            try:
                gs[1].remove()   # text label
            except Exception:
                pass
        self.ghost_spans = []
        self.canvas.draw_idle()


    def _on_aibot_event_found(self, start, end, desc):
        # Store for listening/viewing
        self.discovered_intervals.append((start, end, desc))
        
        span = self.ax.axvspan(start, end, color='purple', alpha=0.1)
        label_text = desc.split("|")[0] if "|" in desc else desc
        txt = self.ax.text(start, self.ax.get_ylim()[1] - 0.5, label_text, color='purple', fontsize=8,
                           bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        self.ghost_spans.append((span, txt))
        self.canvas.draw_idle()

    def _on_aibot_detection(self, xmin, xmax, label, weight):
        self._add_annotation_to_tree(xmin, xmax, label, weight=weight, autosave=True)

    def _on_aibot_finished(self, mode):
        self.btn_yamnet.setEnabled(self.btn_enable_aibot.isChecked())
        self.btn_train_expert.setEnabled(self.btn_enable_aibot.isChecked() and len(self.annotations) >= 2)
        self.btn_view_results.setEnabled(bool(self.discovered_intervals))
        
        if mode == 'yamnet_discovery':
            if hasattr(self, 'ai_progress_dialog'):
                self.ai_progress_dialog.hide()
            self._on_aibot_view_results_clicked()
        elif mode == 'learning':
            if hasattr(self, 'ai_progress_dialog'):
                self.ai_progress_dialog.hide()
            QMessageBox.information(self, "AI Learning", "Custom Expert Model has been trained successfully and is now active!")

    def _on_aibot_train_expert_clicked(self):
        """
        Gathers current labels and starts background expert training.
        """
        if self.audio_data is None:
            QMessageBox.warning(self, "AI Error", "Audio data must be loaded.")
            return

        if len(self.annotations) < 2:
            QMessageBox.warning(self, "Insufficient Data", "Please create at least 2 different types of labels to train the expert.")
            return

        self.ai_progress_dialog = AIProgressDialog("Teaching the AI Expert", "Initializing Expert Learning...", self)
        self.ai_progress_dialog.show()

        # Gather relevant data for thread
        window_start = self.ax.get_xlim()[0]
        window_end = self.ax.get_xlim()[1]

        self.current_aibot_thread = AIBotInferenceThread(
            mode='learning',
            filepath=self.filepath,
            sr=self.sample_rate,
            n_channels=self.n_channels,
            audio_data=self.audio_data,
            spec_data=None, # Not needed for YAMNet
            spec_settings=None,
            window_xlim=(window_start, window_end)
        )
        
        # Pass the current annotations to the thread
        self.current_aibot_thread.training_annotations = [
            {'label': ann['label'], 'start': ann['xmin'], 'end': ann['xmax']} for ann in self.annotations
        ]

        self.current_aibot_thread.progress.connect(self._on_aibot_progress)
        self.current_aibot_thread.log_signal.connect(self._on_aibot_log)
        self.current_aibot_thread.error.connect(self._on_aibot_error)
        self.current_aibot_thread.finished.connect(self._on_aibot_finished)
        self.current_aibot_thread.start()
        
        self.btn_train_expert.setEnabled(False)

    def _on_aibot_reset_expert_clicked(self):
        reply = QMessageBox.question(self, 'Reset AI', "Are you sure you want to delete the custom expert model?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            path = get_resource_path('custom_expert.joblib')
            if os.path.exists(path):
                try:
                    os.remove(path)
                    QMessageBox.information(self, "AI Reset", "Custom Expert Model has been deleted.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not delete model: {e}")
            else:
                QMessageBox.information(self, "AI Reset", "No custom model found to delete.")

    def _on_aibot_error(self, err):
        self.btn_yamnet.setEnabled(True)
        QMessageBox.critical(self, "AI Bot", f"AI Error: {err}")

    def export_data(self):
        if not self.annotations:
            QMessageBox.information(self, "Empty Dataset", "No timeline annotations exist!")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Spreadsheet", "",
            "Excel Workbook (*.xlsx);;CSV Formatting (*.csv)")
        if not filepath:
            return
        df = pd.DataFrame(self.annotations)[["start", "end", "label"]]
        df.rename(columns={
            "start": "Absolute Tracking Start",
            "end":   "Absolute Tracking End",
            "label": "Class ID"
        }, inplace=True)
        try:
            if filepath.endswith('.xlsx'):
                df.to_excel(filepath, index=False)
            else:
                df.to_csv(filepath, index=False)
            QMessageBox.information(
                self, "Success",
                f"Timestamps exported to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                                 f"Could not write output:\n{e}")

    def closeEvent(self, event):
        """Cleanly stop background threads and timers on exit."""
        if hasattr(self, 'res_monitor'):
            self.res_monitor.stop()
        super().closeEvent(event)

    def _on_resource_stats(self, cpu, ram_avail, ram_total):
        """Update internal state and status bar with latest hardware stats."""
        self.latest_resources = (cpu, ram_avail, ram_total)
        
        mem_used_pct = ((ram_total - ram_avail) / ram_total) * 100
        status_text = f"[RAM: {mem_used_pct:.0f}%] [CPU: {cpu:.0f}%]"
        
        if hasattr(self, 'lbl_resource_status'):
            self.lbl_resource_status.setText(status_text)
            
            # Visual alerts based on safety tiers
            if ram_avail < 1.0: # Red < 1GB
                self.lbl_resource_status.setStyleSheet("color: white; background-color: #E91E63; border-radius: 3px; padding: 2px;")
            elif ram_avail < 2.0: # Yellow < 2GB
                self.lbl_resource_status.setStyleSheet("color: black; background-color: #FFC107; border-radius: 3px; padding: 2px;")
            else: # Green
                self.lbl_resource_status.setStyleSheet("color: #4CAF50; font-weight: bold;")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            'esg.amms.annotator.2')
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(ICON_PATH))
    app.setStyle("Fusion")

    window = AudioAnnotatorApp()
    window.showMaximized()
    sys.exit(app.exec_())
