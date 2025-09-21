import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QTabWidget
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
import numpy as np
import librosa
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
        self.animation_timer.setInterval(30) # ~33 FPS
        self.animation_timer.timeout.connect(self._update_plot_animation)

        # --- Plotting objects ---
        self.plot_cache = {}
        
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
        self.stop_playback() # This also redraws the static plot

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
                self.select_plot_type('waveform') # Default to waveform
            except Exception as e:
                error_msg = str(e)
                if "shared object file" in error_msg or "backend" in error_msg:
                    detailed_error = (
                        "Failed to load the audio file due to a missing system library.\n\n"
                        "This is often caused by a missing FFmpeg installation, which is required for loading compressed audio formats like MP3.\n\n"
                        "To fix this, please install FFmpeg on your system and ensure it is available in your system's PATH.\n\n"
                        f"Original error: {error_msg}"
                    )
                    QMessageBox.critical(self, "Missing Dependency", detailed_error)
                else:
                    QMessageBox.critical(self, "Error", f"Could not open or read the audio file.\n{error_msg}")
                self.reset_app_state()

    def load_audio(self):
        if self.filepath:
            self.audio_data, self.sample_rate = librosa.load(self.filepath, sr=None, mono=True)
            self.plot_x_data = np.linspace(0, len(self.audio_data) / self.sample_rate, num=len(self.audio_data))

    def reset_app_state(self):
        # ... (similar logic to reset UI state)
        self.filepath = None
        self.audio_data = None
        self.sample_rate = None
        self.file_label.setText("No file selected")
        for btn in self.buttons.values(): btn.setEnabled(False)
        self.play_pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.plot_widget.clear()


    # --- Plotting Logic ---
    def _draw_static_plot(self):
        self.plot_widget.clear()
        
        if self.active_plot_type == 'waveform':
            self.plot_widget.plot(self.plot_x_data, self.audio_data, pen='k')
            self.plot_widget.setTitle("Audio Waveform")
            self.plot_widget.setLabel('bottom', "Time (s)")
            self.plot_widget.setLabel('left', "Amplitude")
        elif self.active_plot_type in ['spectrum', 'psd']:
            # For static view, we show the full analysis
            if self.active_plot_type == 'spectrum':
                n = len(self.audio_data)
                yf = np.fft.fft(self.audio_data)
                xf = np.fft.fftfreq(n, 1/self.sample_rate)[:n//2]
                y_data = 2.0/n * np.abs(yf[0:n//2])
                self.plot_widget.plot(xf, y_data, pen='k')
                self.plot_widget.setTitle("Frequency Spectrum (Entire Audio)")
                self.plot_widget.setLabel('bottom', "Frequency (Hz)")
            else: # PSD
                # Using matplotlib for the PSD calculation as it's convenient
                import matplotlib.pyplot as plt
                psd_data, freqs = plt.psd(self.audio_data, Fs=self.sample_rate, NFFT=self.n_fft)
                plt.close() # Close the matplotlib figure
                self.plot_widget.plot(freqs, 10 * np.log10(psd_data), pen='k')
                self.plot_widget.setTitle("Power Spectral Density (Entire Audio)")
                self.plot_widget.setLabel('bottom', "Frequency (Hz)")
                self.plot_widget.setLabel('left', "Power/Frequency (dB/Hz)")
        else: # 2D plots
            data = None
            if self.active_plot_type == 'spectrogram':
                data = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data, n_fft=self.n_fft)), ref=np.max)
            elif self.active_plot_type == 'chromagram':
                data = librosa.feature.chroma_stft(y=self.audio_data, sr=self.sample_rate, n_fft=self.n_fft)
            elif self.active_plot_type == 'tempogram':
                data = librosa.feature.tempogram(onset_envelope=librosa.onset.onset_strength(y=self.audio_data, sr=self.sample_rate), sr=self.sample_rate)
            elif self.active_plot_type == 'tonnetz':
                data = librosa.feature.tonnetz(y=librosa.effects.harmonic(self.audio_data), sr=self.sample_rate)

            if data is not None:
                img = pg.ImageItem(image=data.T) # PyQtGraph expects (width, height)
                self.plot_widget.addItem(img)
                # Set correct scaling for the image
                duration = len(self.audio_data) / self.sample_rate
                img.setRect(0, 0, duration, data.shape[0])
                self.plot_widget.setAspectLocked(False)
                self.plot_widget.setTitle(f"{self.active_plot_type.capitalize()} (Entire Audio)")
                self.plot_widget.setLabel('bottom', "Time (s)")

    # --- Playback and Animation ---
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
            # Use a thread-safe signal to stop playback from the callback
            self.main_layout.parentWidget().parent().stop_playback()
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
        
        if self.active_plot_type == 'waveform':
            self.plot_widget.plot(self.plot_x_data, self.audio_data, pen=pg.mkPen(color=(200, 200, 200)))
            self.plot_item = self.plot_widget.plot([], [], pen=pg.mkPen(color='b', width=2))
        elif self.active_plot_type in ['spectrum', 'psd']:
            if self.active_plot_type == 'spectrum':
                xf = np.fft.fftfreq(self.n_fft, 1/self.sample_rate)[:self.n_fft//2]
                self.plot_item = self.plot_widget.plot(xf, np.zeros(self.n_fft//2), pen='b')
                self.plot_widget.setLogMode(x=False, y=False)
                self.plot_widget.setYRange(0, 0.1) # Educated guess for starting range
            else: #PSD
                xf = np.fft.fftfreq(self.n_fft, 1/self.sample_rate)[:self.n_fft//2]
                self.plot_item = self.plot_widget.plot(xf, np.zeros(self.n_fft//2), pen='b')
                self.plot_widget.setYRange(-100, 20)
        else: # Reveal animation for all 2D plots
            self._draw_static_plot() # Draw the full plot first
            # We will update the image data directly in the animation loop
            if self.plot_widget.allChildItems():
                img_item = self.plot_widget.allChildItems()[0]
                if isinstance(img_item, pg.ImageItem):
                    original_data = img_item.image.copy()
                    animation_data = np.full(original_data.shape, np.nanmin(original_data))
                    img_item.setImage(animation_data)
                    self.plot_cache['image_artist'] = img_item
                    self.plot_cache['original_data'] = original_data

    def _update_plot_animation(self):
        if not self.is_playing or self.is_paused: return

        if self.active_plot_type == 'waveform':
            self.plot_item.setData(self.plot_x_data[:self.current_frame], self.audio_data[:self.current_frame])
        elif self.active_plot_type in ['spectrum', 'psd']:
             chunk = self.audio_data[self.current_frame - self.n_fft : self.current_frame]
             if len(chunk) == self.n_fft:
                if self.active_plot_type == 'spectrum':
                    yf = np.fft.fft(chunk)
                    y_data = 2.0/self.n_fft * np.abs(yf[0:self.n_fft//2])
                else: #PSD
                    window = np.hanning(self.n_fft)
                    chunk = chunk * window
                    yf = np.fft.fft(chunk)
                    psd_data = (np.abs(yf)**2) / (self.sample_rate * np.sum(window**2))
                    y_data = 10 * np.log10(psd_data[:self.n_fft//2])
                    y_data[np.isneginf(y_data)] = -120
                self.plot_item.setData(y=y_data)
        else: # Reveal animation
            if 'image_artist' in self.plot_cache:
                img = self.plot_cache['image_artist']
                original = self.plot_cache['original_data']
                current_time = self.current_frame / self.sample_rate
                duration = len(self.audio_data) / self.sample_rate
                if duration > 0:
                    cols_to_show = int((current_time / duration) * original.shape[1])
                    if cols_to_show > 0:
                        new_data = img.image
                        new_data[:,:cols_to_show] = original[:,:cols_to_show]
                        img.setImage(new_data, autoLevels=False)


    def closeEvent(self, event):
        self.stop_playback()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = AudioAnalyzerApp()
    main_win.show()
    sys.exit(app.exec())