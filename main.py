import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display
import sounddevice as sd
import threading
import time

class AudioAnalyzerApp:
    """
    A Python application to analyze and visualize audio files with real-time playback and plotting.
    """
    def __init__(self, master):
        """
        Initializes the Audio Analyzer application.
        """
        self.master = master
        self.master.title("Audio Analyzer")
        self.master.geometry("950x750")

        self.filepath = None
        self.audio_data = None
        self.sample_rate = None
        self.active_plot_func = self.plot_waveform # Keep track of the current plot

        # --- Playback and Animation control ---
        self.playback_thread = None
        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0
        self.animation_job = None

        # --- Plotting objects ---
        self.playhead_line = None
        self.dynamic_line = None
        self.plot_x_data = None
        self.plot_y_data = None
        
        # --- UI Elements ---
        self.main_frame = tk.Frame(self.master, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # File selection
        self.file_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.file_frame.pack(pady=10)
        self.select_button = tk.Button(self.file_frame, text="Select Audio File", command=self.select_file, font=("Helvetica", 12))
        self.select_button.pack(side=tk.LEFT, padx=5)
        self.file_label = tk.Label(self.file_frame, text="No file selected", font=("Helvetica", 10), bg="#f0f0f0", width=60, anchor='w')
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        # --- Playback controls ---
        self.playback_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.playback_frame.pack(pady=5)
        self.play_pause_button = tk.Button(self.playback_frame, text="▶ Play", command=self.toggle_playback, state=tk.DISABLED, font=("Helvetica", 12), width=8)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(self.playback_frame, text="■ Stop", command=self.stop_playback, state=tk.DISABLED, font=("Helvetica", 12), width=8)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # --- Tabbed interface for graphs ---
        self.tab_control = ttk.Notebook(self.main_frame)
        self.tab2d = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab2d, text='2D Graphs')
        self.tab_control.pack(expand=1, fill="x", pady=5)

        # --- Analysis buttons (inside the 2D tab) ---
        self.analysis_frame_container = tk.Frame(self.tab2d, bg="#f0f0f0")
        self.analysis_frame_container.pack(pady=5)
        self.analysis_frame1 = tk.Frame(self.analysis_frame_container, bg="#f0f0f0")
        self.analysis_frame1.pack()
        self.waveform_button = tk.Button(self.analysis_frame1, text="Waveform", command=self.plot_waveform, state=tk.DISABLED, font=("Helvetica", 12))
        self.waveform_button.pack(side=tk.LEFT, padx=5, pady=2)
        self.spectrum_button = tk.Button(self.analysis_frame1, text="Frequency Spectrum", command=self.plot_spectrum, state=tk.DISABLED, font=("Helvetica", 12))
        self.spectrum_button.pack(side=tk.LEFT, padx=5, pady=2)
        self.psd_button = tk.Button(self.analysis_frame1, text="Power Spectral Density", command=self.plot_psd, state=tk.DISABLED, font=("Helvetica", 12))
        self.psd_button.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.analysis_frame2 = tk.Frame(self.analysis_frame_container, bg="#f0f0f0")
        self.analysis_frame2.pack()
        self.spectrogram_button = tk.Button(self.analysis_frame2, text="Spectrogram", command=self.plot_spectrogram, state=tk.DISABLED, font=("Helvetica", 12))
        self.spectrogram_button.pack(side=tk.LEFT, padx=5, pady=2)
        self.chromagram_button = tk.Button(self.analysis_frame2, text="Chromagram", command=self.plot_chromagram, state=tk.DISABLED, font=("Helvetica", 12))
        self.chromagram_button.pack(side=tk.LEFT, padx=5, pady=2)
        self.tempogram_button = tk.Button(self.analysis_frame2, text="Tempogram", command=self.plot_tempogram, state=tk.DISABLED, font=("Helvetica", 12))
        self.tempogram_button.pack(side=tk.LEFT, padx=5, pady=2)
        self.tonnetz_button = tk.Button(self.analysis_frame2, text="Tonnetz", command=self.plot_tonnetz, state=tk.DISABLED, font=("Helvetica", 12))
        self.tonnetz_button.pack(side=tk.LEFT, padx=5, pady=2)

        # Plotting canvas
        self.plot_frame = tk.Frame(self.main_frame, bg="white", relief=tk.SUNKEN, borderwidth=1)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.fig = plt.Figure(figsize=(10, 6), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def select_file(self):
        self.stop_playback()
        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(("Audio Files", "*.wav *.mp3 *.flac"), ("All files", "*.*"))
        )
        if filepath:
            try:
                self.filepath = filepath
                self.file_label.config(text=self.filepath.split('/')[-1])
                self.load_audio()
                for button in [self.waveform_button, self.spectrum_button, self.psd_button, self.spectrogram_button, self.chromagram_button, self.tempogram_button, self.tonnetz_button, self.play_pause_button, self.stop_button]:
                    button.config(state=tk.NORMAL)
                self.plot_waveform() 
            except Exception as e:
                messagebox.showerror("Error", f"Could not open or read the audio file.\n{e}")
                self.reset_app_state()

    def load_audio(self):
        if self.filepath:
            self.audio_data, self.sample_rate = librosa.load(self.filepath, sr=None, mono=True)
            self.plot_x_data = np.linspace(0, len(self.audio_data) / self.sample_rate, num=len(self.audio_data))
            self.plot_y_data = self.audio_data

    def reset_app_state(self):
        self.filepath = None
        self.audio_data = None
        self.sample_rate = None
        self.file_label.config(text="No file selected")
        for button in [self.waveform_button, self.spectrum_button, self.psd_button, self.spectrogram_button, self.chromagram_button, self.tempogram_button, self.tonnetz_button, self.play_pause_button, self.stop_button]:
            button.config(state=tk.DISABLED)
        self.clear_plot()
    
    def clear_plot(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.playhead_line = None
        self.dynamic_line = None
        self.canvas.draw()
        
    def _draw_static_plot(self, plot_function, *args, **kwargs):
        """Template for drawing a full, static plot."""
        self._internal_stop_playback_only()
        self.clear_plot()
        plot_function(*args, **kwargs)
        self.canvas.draw()
        
    def plot_waveform(self):
        if self.audio_data is not None:
            self.active_plot_func = self.plot_waveform
            self._draw_static_plot(lambda: (
                librosa.display.waveshow(y=self.audio_data, sr=self.sample_rate, ax=self.ax),
                self.ax.set_title("Audio Waveform"),
                self.ax.set_xlabel("Time (s)"),
                self.ax.set_ylabel("Amplitude")
            ))

    def plot_spectrum(self):
        messagebox.showinfo("Playback Info", "Real-time playback is not applicable to frequency-domain graphs. The full spectrum will be shown.")
        if self.audio_data is not None:
            self.active_plot_func = self.plot_spectrum
            self._draw_static_plot(lambda: (
                n := len(self.audio_data),
                T := 1.0 / self.sample_rate,
                yf := np.fft.fft(self.audio_data),
                xf := np.fft.fftfreq(n, T)[:n//2],
                self.ax.plot(xf, 2.0/n * np.abs(yf[0:n//2])),
                self.ax.set_title("Frequency Spectrum"),
                self.ax.set_xlabel("Frequency (Hz)"),
                self.ax.set_ylabel("Amplitude"),
                self.ax.grid()
            ))

    def plot_psd(self):
        messagebox.showinfo("Playback Info", "Real-time playback is not applicable to frequency-domain graphs. The full Power Spectral Density will be shown.")
        if self.audio_data is not None:
            self.active_plot_func = self.plot_psd
            self._draw_static_plot(lambda: (
                self.ax.psd(self.audio_data, Fs=self.sample_rate, NFFT=2048),
                self.ax.set_title("Power Spectral Density"),
                self.ax.set_xlabel("Frequency (Hz)"),
                self.ax.set_ylabel("Power/Frequency (dB/Hz)")
            ))
            
    def plot_spectrogram(self):
        if self.audio_data is not None:
            self.active_plot_func = self.plot_spectrogram
            self._draw_static_plot(lambda: (
                D := librosa.stft(self.audio_data),
                S_db := librosa.amplitude_to_db(np.abs(D), ref=np.max),
                img := librosa.display.specshow(S_db, sr=self.sample_rate, x_axis='time', y_axis='log', ax=self.ax),
                self.fig.colorbar(img, ax=self.ax, format='%+2.0f dB', label='Intensity'),
                self.ax.set_title('Spectrogram'),
                self.ax.set_xlabel("Time (s)"),
                self.ax.set_ylabel("Frequency (Hz)")
            ))

    def plot_chromagram(self):
        if self.audio_data is not None:
            self.active_plot_func = self.plot_chromagram
            self._draw_static_plot(lambda: (
                chroma := librosa.feature.chroma_stft(y=self.audio_data, sr=self.sample_rate),
                img := librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=self.ax, sr=self.sample_rate),
                self.fig.colorbar(img, ax=self.ax, label='Intensity'),
                self.ax.set_title('Chromagram'),
                self.ax.set_xlabel("Time (s)"),
                self.ax.set_ylabel("Pitch Class")
            ))

    def plot_tempogram(self):
        if self.audio_data is not None:
            self.active_plot_func = self.plot_tempogram
            self._draw_static_plot(lambda: (
                onset_env := librosa.onset.onset_strength(y=self.audio_data, sr=self.sample_rate),
                tempogram := librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sample_rate),
                img := librosa.display.specshow(tempogram, sr=self.sample_rate, x_axis='time', y_axis='tempo', ax=self.ax, cmap='magma'),
                self.fig.colorbar(img, ax=self.ax, label='Strength'),
                self.ax.set_title('Tempogram'),
                self.ax.set_xlabel("Time (s)"),
                self.ax.set_ylabel("Tempo (BPM)")
            ))

    def plot_tonnetz(self):
        if self.audio_data is not None:
            self.active_plot_func = self.plot_tonnetz
            self._draw_static_plot(lambda: (
                tonnetz := librosa.feature.tonnetz(y=librosa.effects.harmonic(self.audio_data), sr=self.sample_rate),
                img := librosa.display.specshow(tonnetz, sr=self.sample_rate, x_axis='time', y_axis='tonnetz', ax=self.ax, cmap='coolwarm'),
                self.fig.colorbar(img, ax=self.ax),
                self.ax.set_title('Tonal Centroid Network (Tonnetz)'),
                self.ax.set_xlabel("Time (s)"),
                self.ax.set_ylabel("Tonal Centroids")
            ))
            
    def toggle_playback(self):
        if self.is_playing:
            self.is_paused = not self.is_paused
            self.play_pause_button.config(text="▶ Play" if self.is_paused else "❚❚ Pause")
        else:
            self.is_playing = True
            self.is_paused = False
            self.play_pause_button.config(text="❚❚ Pause")
            
            self.prepare_plot_for_animation()

            self.playback_thread = threading.Thread(target=self._audio_playback_loop, daemon=True)
            self.playback_thread.start()
            self._update_plot_animation()

    def _internal_stop_playback_only(self):
        """Stops audio and animation without redrawing the plot."""
        if self.animation_job:
            self.master.after_cancel(self.animation_job)
            self.animation_job = None
        
        if self.is_playing:
            self.is_playing = False
            if self.playback_thread and self.playback_thread.is_alive():
                 self.playback_thread.join(timeout=0.1)
        
        self.current_frame = 0
        self.is_paused = False
        self.play_pause_button.config(text="▶ Play")

    def stop_playback(self):
        """Stops playback and restores the full static plot."""
        self._internal_stop_playback_only()
        if self.audio_data is not None:
            self.active_plot_func()

    def _audio_playback_loop(self):
        try:
            with sd.OutputStream(samplerate=self.sample_rate, channels=1) as stream:
                while self.current_frame < len(self.audio_data) and self.is_playing:
                    if not self.is_paused:
                        remaining = len(self.audio_data) - self.current_frame
                        chunk_size = min(2048, remaining) 
                        chunk = self.audio_data[self.current_frame:self.current_frame + chunk_size]
                        stream.write(chunk.astype('float32'))
                        self.current_frame += chunk_size
                    else:
                        time.sleep(0.05)
        except Exception as e:
             self.master.after(0, lambda: messagebox.showerror("Playback Error", str(e)))

        self.master.after(0, self.stop_playback)

    def prepare_plot_for_animation(self):
        self.clear_plot()
        self.ax.set_xlim(0, len(self.audio_data) / self.sample_rate)
        self.ax.set_ylim(np.min(self.audio_data), np.max(self.audio_data))

        if self.active_plot_func == self.plot_waveform:
            self.ax.set_title("Audio Waveform (Playing...)")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude")
            self.dynamic_line, = self.ax.plot([], [], lw=1)
        else: 
            self.active_plot_func()
            self.playhead_line = self.ax.axvline(x=0, color='r', linestyle='--', zorder=10)

    def _update_plot_animation(self):
        if not self.is_playing or self.is_paused:
            return

        current_time = self.current_frame / self.sample_rate

        if self.active_plot_func == self.plot_waveform and self.dynamic_line:
            self.dynamic_line.set_data(self.plot_x_data[:self.current_frame], self.plot_y_data[:self.current_frame])
        
        elif self.playhead_line:
            self.playhead_line.set_xdata([current_time, current_time])

        self.canvas.draw()
        self.animation_job = self.master.after(40, self._update_plot_animation)

    def on_closing(self):
        self._internal_stop_playback_only()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()

