import sys
import os

# --- Disable Numba for Librosa via Monkey-Patching ---
# This is a more robust method than environment variables. It directly replaces
# the part of librosa that calls numba with a dummy function, preventing crashes.
os.environ['LIBROSA_NO_NUMBA'] = 'True' # Keep as a fallback
import librosa
try:
    # The dummy decorator that will replace numba.jit
    def no_op_decorator(f):
        return f
    
    # Replace the jit decorator in the part of librosa that uses it
    import librosa.core.spectrum
    librosa.core.spectrum.jit = no_op_decorator
    print("Numba disabled for Librosa via monkey-patch.")
except (ImportError, AttributeError):
    print("Could not apply Numba monkey-patch.")
    pass

# --- FFmpeg Path Configuration ---
script_dir = os.path.dirname(os.path.realpath(__file__))
ffmpeg_bin_path = os.path.join(script_dir, "ffmpeg", "bin")
ffmpeg_exe_path = os.path.join(ffmpeg_bin_path, "ffmpeg.exe")
ffprobe_exe_path = os.path.join(ffmpeg_bin_path, "ffprobe.exe")

if os.path.isdir(ffmpeg_bin_path):
    os.environ['PATH'] = f"{ffmpeg_bin_path}{os.pathsep}{os.environ['PATH']}"

from pydub import AudioSegment

if os.path.isfile(ffmpeg_exe_path):
    AudioSegment.converter = ffmpeg_exe_path
if os.path.isfile(ffprobe_exe_path):
    AudioSegment.ffprobe = ffprobe_exe_path

# Now, import the rest of the libraries
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QTabWidget
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
import numpy as np
import sounddevice as sd

# --- PyTorch CUDA Check ---
try:
    import torch
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
except ImportError:
    torch = None
    cuda_available = False
    device = "cpu"


class AudioAnalyzerApp(QMainWindow):
    """
    A Python application to analyze and visualize audio files, rebuilt with PyQt and PyQtGraph for high performance.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Analyzer")
        self.setGeometry(100, 100, 1000, 800)

        self.filepath = None
        self.audio_data = None
        self.sample_rate = None
        self.active_plot_type = 'waveform'
        self.n_fft = 2048

        # --- Playback and Animation control ---
        self.stream = None
        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0
        self.animation_timer = QTimer()
        self.animation_timer.setInterval(25) # ~40 FPS for smoother animation
        self.animation_timer.timeout.connect(self._update_plot_animation)

        # --- Plotting objects ---
        self.plot_cache = {}
        self.animation_item = None
        
        # --- UI Elements ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # --- Top bar: File selection and CUDA status ---
        top_bar_layout = QHBoxLayout()
        self.select_button = QPushButton("Select Audio File")
        self.select_button.clicked.connect(self.select_file)
        self.file_label = QLabel("No file selected")
        top_bar_layout.addWidget(self.select_button)
        top_bar_layout.addWidget(self.file_label, 1) # Stretchable
        
        if cuda_available:
            self.cuda_label = QLabel("CUDA AVAILABLE")
            self.cuda_label.setStyleSheet("color: green; font-weight: bold;")
            top_bar_layout.addWidget(self.cuda_label)
        
        self.main_layout.addLayout(top_bar_layout)

        # --- Playback controls ---
        playback_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("▶ Play")
        self.play_pause_button.setFixedWidth(100)
        self.play_pause_button.setEnabled(False)
        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.stop_button = QPushButton("■ Stop")
        self.stop_button.setFixedWidth(100)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_playback)
        playback_layout.addWidget(self.play_pause_button)
        playback_layout.addWidget(self.stop_button)
        self.main_layout.addLayout(playback_layout)

        # --- Tabbed interface for graphs ---
        self.tab_widget = QTabWidget()
        self.tab2d = QWidget()
        self.tab_widget.addTab(self.tab2d, "2D Graphs")
        self.main_layout.addWidget(self.tab_widget)
        
        tab_layout = QVBoxLayout(self.tab2d)
        analysis_layout1 = QHBoxLayout()
        analysis_layout2 = QHBoxLayout()

        self.buttons = {}
        plot_types = {
            'Waveform': ('waveform', analysis_layout1), 'Frequency Spectrum': ('spectrum', analysis_layout1), 'Power Spectral Density': ('psd', analysis_layout1),
            'Spectrogram': ('spectrogram', analysis_layout2), 'Chromagram': ('chromagram', analysis_layout2), 'Tempogram': ('tempogram', analysis_layout2), 'Tonnetz': ('tonnetz', analysis_layout2)
        }
        for name, (ptype, layout) in plot_types.items():
            btn = QPushButton(name)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, p=ptype: self.select_plot_type(p))
            layout.addWidget(btn)
            self.buttons[ptype] = btn
        
        tab_layout.addLayout(analysis_layout1)
        tab_layout.addLayout(analysis_layout2)
        
        # --- PyQtGraph Plotting Widget ---
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.plot_widget = pg.PlotWidget()
        self.main_layout.addWidget(self.plot_widget)

    def select_plot_type(self, ptype):
        self.active_plot_type = ptype
        self.stop_playback()

    def select_file(self):
        self.stop_playback()
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.flac)")
        if filepath:
            try:
                self.filepath = filepath
                self.file_label.setText(filepath.split('/')[-1])
                self.plot_cache.clear()
                self.load_audio()
                for btn in self.buttons.values(): btn.setEnabled(True)
                self.play_pause_button.setEnabled(True)
                self.stop_button.setEnabled(True)
                self.select_plot_type('waveform')
            except Exception as e:
                error_msg = str(e)
                if "ffmpeg" in error_msg.lower() or "could not read file" in error_msg.lower():
                    detailed_error = (
                        "Failed to load audio: Missing or incorrect FFmpeg setup.\n\n"
                        "This application requires FFmpeg to load compressed audio formats like MP3.\n\n"
                        "To fix this:\n"
                        "1. Download FFmpeg (from ffmpeg.org).\n"
                        "2. Extract the downloaded archive.\n"
                        "3. Rename the extracted folder to exactly 'ffmpeg'.\n"
                        "4. Place the 'ffmpeg' folder in the same directory as this application.\n\n"
                        f"Original error: {error_msg}"
                    )
                    QMessageBox.critical(self, "Missing Dependency", detailed_error)
                else:
                    QMessageBox.critical(self, "Error", f"Could not open or read the audio file.\n{error_msg}")
                self.reset_app_state()

    def load_audio(self):
        if self.filepath:
            audio_segment = AudioSegment.from_file(self.filepath)
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            self.audio_data = samples / (2**(audio_segment.sample_width * 8 - 1))
            self.sample_rate = audio_segment.frame_rate
            self.plot_x_data = np.linspace(0, len(self.audio_data) / self.sample_rate, num=len(self.audio_data))

    def reset_app_state(self):
        self.filepath = None
        self.audio_data = None
        self.sample_rate = None
        self.file_label.setText("No file selected")
        for btn in self.buttons.values(): btn.setEnabled(False)
        self.play_pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.plot_widget.clear()

    def _get_plot_data(self, plot_key, calculation_func):
        if plot_key not in self.plot_cache:
            self.plot_cache[plot_key] = calculation_func()
        return self.plot_cache[plot_key]

    def _draw_static_plot(self):
        self.plot_widget.clear()
        
        plot_function = getattr(self, f"_draw_{self.active_plot_type}_on_ax", None)
        if callable(plot_function):
            plot_function()
        
        self.plot_widget.autoRange()

    def _draw_waveform_on_ax(self):
        self.plot_widget.getPlotItem().setLogMode(x=False, y=False)
        plot_item = self.plot_widget.plot(self.plot_x_data, self.audio_data, pen='k')
        plot_item.setDownsampling(auto=True, method='peak')
        self.plot_widget.setTitle("Audio Waveform")
        self.plot_widget.setLabel('bottom', "Time (s)")
        self.plot_widget.setLabel('left', "Amplitude")

    def _draw_spectrum_on_ax(self):
        n = len(self.audio_data)
        yf = np.fft.fft(self.audio_data)
        xf = np.fft.fftfreq(n, 1/self.sample_rate)[:n//2]
        y_data_linear = 2.0/n * np.abs(yf[0:n//2])
        y_data_db = 20 * np.log10(y_data_linear + 1e-9)
        
        self.plot_widget.getPlotItem().setLogMode(x=True, y=False)
        self.plot_widget.plot(xf, y_data_db, pen='k')
        self.plot_widget.setTitle("Frequency Spectrum (Entire Audio)")
        self.plot_widget.setLabel('bottom', "Frequency (Hz)")
        self.plot_widget.setLabel('left', "Amplitude (dB)")

    def _draw_psd_on_ax(self):
        n = len(self.audio_data)
        yf = np.fft.fft(self.audio_data)
        psd_data = (np.abs(yf[0:n//2])**2) / (self.sample_rate * n)
        freqs = np.fft.fftfreq(n, 1/self.sample_rate)[:n//2]
        
        self.plot_widget.getPlotItem().setLogMode(x=True, y=False)
        self.plot_widget.plot(freqs, 10 * np.log10(psd_data + 1e-9), pen='k')
        self.plot_widget.setTitle("Power Spectral Density (Entire Audio)")
        self.plot_widget.setLabel('bottom', "Frequency (Hz)")
        self.plot_widget.setLabel('left', "Power/Frequency (dB/Hz)")

    def _draw_spectrogram_on_ax(self):
        data = self._get_plot_data('spectrogram', lambda: librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data, n_fft=self.n_fft)), ref=np.max))
        img = pg.ImageItem(image=data.T)
        self.plot_widget.addItem(img)
        duration = len(self.audio_data) / self.sample_rate
        img.setRect(0, 0, duration, data.shape[0])
        self.plot_widget.setAspectLocked(False)
        self.plot_widget.setTitle(f"Spectrogram")
        self.plot_widget.setLabel('bottom', "Time (s)")

    def _draw_chromagram_on_ax(self):
        data = self._get_plot_data('chromagram', lambda: librosa.feature.chroma_stft(y=self.audio_data, sr=self.sample_rate, n_fft=self.n_fft))
        img = pg.ImageItem(image=data.T)
        self.plot_widget.addItem(img)
        duration = len(self.audio_data) / self.sample_rate
        img.setRect(0, 0, duration, data.shape[0])
        self.plot_widget.setAspectLocked(False)
        self.plot_widget.setTitle(f"Chromagram")
        self.plot_widget.setLabel('bottom', "Time (s)")

    def _draw_tempogram_on_ax(self):
        data = self._get_plot_data('tempogram', lambda: librosa.feature.tempogram(onset_envelope=librosa.onset.onset_strength(y=self.audio_data, sr=self.sample_rate), sr=self.sample_rate))
        img = pg.ImageItem(image=data.T)
        self.plot_widget.addItem(img)
        duration = len(self.audio_data) / self.sample_rate
        img.setRect(0, 0, duration, data.shape[0])
        self.plot_widget.setAspectLocked(False)
        self.plot_widget.setTitle(f"Tempogram")
        self.plot_widget.setLabel('bottom', "Time (s)")

    def _draw_tonnetz_on_ax(self):
        data = self._get_plot_data('tonnetz', lambda: librosa.feature.tonnetz(y=librosa.effects.harmonic(self.audio_data), sr=self.sample_rate))
        img = pg.ImageItem(image=data.T)
        self.plot_widget.addItem(img)
        duration = len(self.audio_data) / self.sample_rate
        img.setRect(0, 0, duration, data.shape[0])
        self.plot_widget.setAspectLocked(False)
        self.plot_widget.setTitle(f"Tonnetz")
        self.plot_widget.setLabel('bottom', "Time (s)")

    def _audio_callback(self, outdata, frames, time_info, status):
        if status: print(status, file=sys.stderr)
        if self.is_paused:
            outdata.fill(0)
            return
        chunk_size = min(len(self.audio_data) - self.current_frame, frames)
        chunk = self.audio_data[self.current_frame : self.current_frame + chunk_size]
        outdata[:chunk_size, 0] = chunk
        if chunk_size < frames:
            outdata[chunk_size:, 0] = 0
        self.current_frame += chunk_size

    def toggle_playback(self):
        if not self.is_playing:
            self.is_playing = True
            self.is_paused = False
            self.play_pause_button.setText("❚❚ Pause")
            try:
                self.prepare_plot_for_animation()
                self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, callback=self._audio_callback, blocksize=self.n_fft)
                self.stream.start()
                self.animation_timer.start()
            except Exception as e:
                QMessageBox.critical(self, "Playback Error", f"Could not start audio stream.\n{e}")
                self.stop_playback()
        else:
            self.is_paused = not self.is_paused
            self.play_pause_button.setText("▶ Play" if self.is_paused else "❚❚ Pause")

    def stop_playback(self):
        if self.stream:
            self.stream.stop(ignore_errors=True)
            self.stream.close(ignore_errors=True)
            self.stream = None
        self.animation_timer.stop()
        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0
        self.play_pause_button.setText("▶ Play")
        if self.audio_data is not None:
            self._draw_static_plot()

    def prepare_plot_for_animation(self):
        self.plot_widget.clear()
        
        is_live_freq_plot = self.active_plot_type in ['spectrum', 'psd']
        is_2d_time_plot = self.active_plot_type in ['spectrogram', 'chromagram', 'tempogram', 'tonnetz']

        if self.active_plot_type == 'waveform' or is_2d_time_plot:
            # FIX: Use a playhead for all 2D time plots for performance and consistency
            plot_function = getattr(self, f"_draw_{self.active_plot_type}_on_ax")
            plot_function()
            self.plot_widget.setTitle(f"{self.active_plot_type.capitalize()} (Playing...)")
            self.animation_item = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=2))
            self.plot_widget.addItem(self.animation_item)
            self.animation_item.setPos(0)
            
        elif is_live_freq_plot:
            xf = np.fft.fftfreq(self.n_fft, 1/self.sample_rate)[:self.n_fft//2]
            self.animation_item = self.plot_widget.plot(xf, np.zeros(self.n_fft//2), pen='b')
            self.plot_widget.getPlotItem().setLogMode(x=True, y=False)
            
            if self.active_plot_type == 'spectrum':
                self.plot_widget.setTitle("Live Frequency Spectrum")
                self.plot_widget.setLabel('left', "Amplitude (dB)")
                self.plot_widget.setYRange(-60, 20)
            else: #PSD
                self.plot_widget.setTitle("Live Power Spectral Density")
                self.plot_widget.setLabel('left', "Power/Frequency (dB/Hz)")
                self.plot_widget.setYRange(-100, 20)

    def _update_plot_animation(self):
        if not self.is_playing or self.is_paused: return

        if self.current_frame >= len(self.audio_data):
            self.stop_playback()
            return
            
        current_time = self.current_frame / self.sample_rate if self.sample_rate > 0 else 0
        
        is_2d_time_plot = self.active_plot_type in ['spectrogram', 'chromagram', 'tempogram', 'tonnetz']

        if self.active_plot_type == 'waveform' or is_2d_time_plot:
            self.animation_item.setPos(current_time)
        elif self.active_plot_type in ['spectrum', 'psd']:
             chunk = self.audio_data[self.current_frame - self.n_fft : self.current_frame]
             if len(chunk) == self.n_fft:
                yf = np.fft.fft(chunk)
                if self.active_plot_type == 'spectrum':
                    y_data_linear = 2.0/self.n_fft * np.abs(yf[0:self.n_fft//2])
                    y_data = 20 * np.log10(y_data_linear + 1e-9)
                else: #PSD
                    window = np.hanning(self.n_fft)
                    chunk = chunk * window
                    yf = np.fft.fft(chunk)
                    psd_data = (np.abs(yf)**2) / (self.sample_rate * np.sum(window**2))
                    y_data = 10 * np.log10(psd_data[:self.n_fft//2] + 1e-9)
                y_data[np.isneginf(y_data)] = -120
                self.animation_item.setData(y=y_data)

    def closeEvent(self, event):
        self.stop_playback()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = AudioAnalyzerApp()
    main_win.show()
    sys.exit(app.exec())