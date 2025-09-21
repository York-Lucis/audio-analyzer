import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display
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
        self.active_plot_func = self.plot_waveform
        self.n_fft = 2048 # Window size for live analysis

        # --- Playback and Animation control ---
        self.stream = None
        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0
        self.animation_job = None

        # --- Plotting objects ---
        self.dynamic_line = None
        self.plot_x_data = None
        self.plot_y_data = None
        self.plot_cache = {}
        
        # --- UI Elements ---
        self.main_frame = tk.Frame(self.master, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # File selection and CUDA status
        self.top_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.top_frame.pack(pady=10, fill=tk.X)
        self.select_button = tk.Button(self.top_frame, text="Select Audio File", command=self.select_file, font=("Helvetica", 12))
        self.select_button.pack(side=tk.LEFT, padx=5)
        self.file_label = tk.Label(self.top_frame, text="No file selected", font=("Helvetica", 10), bg="#f0f0f0", width=60, anchor='w')
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        if cuda_available:
            self.cuda_label = tk.Label(self.top_frame, text="CUDA AVAILABLE", font=("Helvetica", 10, "bold"), fg="green", bg="#f0f0f0")
            self.cuda_label.pack(side=tk.RIGHT, padx=10)
        
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
                self.plot_cache.clear()
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
        self.dynamic_line = None
        self.canvas.draw()
        
    def _get_plot_data(self, plot_key, calculation_func):
        if plot_key not in self.plot_cache:
            self.plot_cache[plot_key] = calculation_func()
        return self.plot_cache[plot_key]

    def _draw_static_plot(self, plot_function):
        self._internal_stop_playback_only()
        self.clear_plot()
        plot_function()
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
        if self.audio_data is not None:
            self.active_plot_func = self.plot_spectrum
            self._draw_static_plot(lambda: (
                n := len(self.audio_data), T := 1.0 / self.sample_rate,
                yf := np.fft.fft(self.audio_data), 
                xf := np.fft.fftfreq(n, T)[:n//2],
                y_data := 2.0/n * np.abs(yf[0:n//2]),
                self.plot_cache.update({'spectrum_ymax': np.max(y_data) * 1.1}),
                self.ax.plot(xf, y_data),
                self.ax.set_title("Frequency Spectrum (Entire Audio)"), self.ax.set_xlabel("Frequency (Hz)"),
                self.ax.set_ylabel("Amplitude"), self.ax.grid()
            ))

    def plot_psd(self):
        if self.audio_data is not None:
            self.active_plot_func = self.plot_psd
            def do_plot():
                psd_data, freqs = self.ax.psd(self.audio_data, Fs=self.sample_rate, NFFT=self.n_fft)
                self.plot_cache.update({'psd_ymax': np.max(psd_data), 'psd_ymin': np.min(psd_data)})
                self.ax.set_title("Power Spectral Density (Entire Audio)")
                self.ax.set_xlabel("Frequency (Hz)")
                self.ax.set_ylabel("Power/Frequency (dB/Hz)")
            self._draw_static_plot(do_plot)
            
    def plot_spectrogram(self):
        if self.audio_data is not None:
            self.active_plot_func = self.plot_spectrogram
            data = self._get_plot_data('spectrogram', lambda: librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data)), ref=np.max))
            self._draw_static_plot(lambda: (
                img := librosa.display.specshow(data, sr=self.sample_rate, x_axis='time', y_axis='log', ax=self.ax),
                self.fig.colorbar(img, ax=self.ax, format='%+2.0f dB', label='Intensity'),
                self.ax.set_title('Spectrogram'), self.ax.set_xlabel("Time (s)"), self.ax.set_ylabel("Frequency (Hz)")
            ))

    def plot_chromagram(self):
        if self.audio_data is not None:
            self.active_plot_func = self.plot_chromagram
            data = self._get_plot_data('chromagram', lambda: librosa.feature.chroma_stft(y=self.audio_data, sr=self.sample_rate))
            self._draw_static_plot(lambda: (
                img := librosa.display.specshow(data, y_axis='chroma', x_axis='time', ax=self.ax, sr=self.sample_rate),
                self.fig.colorbar(img, ax=self.ax, label='Intensity'),
                self.ax.set_title('Chromagram'), self.ax.set_xlabel("Time (s)"), self.ax.set_ylabel("Pitch Class")
            ))

    def plot_tempogram(self):
        if self.audio_data is not None:
            self.active_plot_func = self.plot_tempogram
            data = self._get_plot_data('tempogram', lambda: librosa.feature.tempogram(onset_envelope=librosa.onset.onset_strength(y=self.audio_data, sr=self.sample_rate), sr=self.sample_rate))
            self._draw_static_plot(lambda: (
                img := librosa.display.specshow(data, sr=self.sample_rate, x_axis='time', y_axis='tempo', ax=self.ax, cmap='magma'),
                self.fig.colorbar(img, ax=self.ax, label='Strength'),
                self.ax.set_title('Tempogram'), self.ax.set_xlabel("Time (s)"), self.ax.set_ylabel("Tempo (BPM)")
            ))

    def plot_tonnetz(self):
        if self.audio_data is not None:
            self.active_plot_func = self.plot_tonnetz
            data = self._get_plot_data('tonnetz', lambda: librosa.feature.tonnetz(y=librosa.effects.harmonic(self.audio_data), sr=self.sample_rate))
            self._draw_static_plot(lambda: (
                img := librosa.display.specshow(data, sr=self.sample_rate, x_axis='time', y_axis='tonnetz', ax=self.ax, cmap='coolwarm'),
                self.fig.colorbar(img, ax=self.ax),
                self.ax.set_title('Tonal Centroid Network (Tonnetz)'),
                self.ax.set_xlabel("Time (s)"), self.ax.set_ylabel("Tonal Centroids")
            ))
            
    def _audio_callback(self, outdata, frames, time, status):
        if status:
            # Don't print underflow errors to avoid console spam
            if status != sd.CallbackFlags.output_underflow:
                print(status)
        
        if self.is_paused:
            outdata.fill(0)
            return

        chunk_size = min(len(self.audio_data) - self.current_frame, frames)
        chunk = self.audio_data[self.current_frame : self.current_frame + chunk_size]
        
        outdata[:chunk_size] = chunk.reshape(-1, 1)
        
        if chunk_size < frames:
            outdata[chunk_size:] = 0
            self.master.after(0, self.stop_playback)
        
        self.current_frame += chunk_size

    def toggle_playback(self):
        if not self.is_playing:
            self.is_playing = True
            self.is_paused = False
            self.play_pause_button.config(text="❚❚ Pause")
            
            try:
                self.prepare_plot_for_animation()
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=self._audio_callback,
                    blocksize=self.n_fft
                )
                self.stream.start()
                self._update_plot_animation()
            except Exception as e:
                messagebox.showerror("Playback Error", f"Could not start audio stream.\n{e}")
                self.stop_playback()
        else:
            self.is_paused = not self.is_paused
            self.play_pause_button.config(text="▶ Play" if self.is_paused else "❚❚ Pause")

    def _internal_stop_playback_only(self):
        if self.animation_job:
            self.master.after_cancel(self.animation_job)
            self.animation_job = None
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0
        self.play_pause_button.config(text="▶ Play")

    def stop_playback(self):
        self._internal_stop_playback_only()
        if 'clip_rect' in self.plot_cache: del self.plot_cache['clip_rect']
        if self.audio_data is not None: self.active_plot_func()

    def prepare_plot_for_animation(self):
        is_waveform = self.active_plot_func == self.plot_waveform
        is_live_freq = self.active_plot_func in [self.plot_spectrum, self.plot_psd]

        self.clear_plot()
        if is_waveform:
            self.ax.set_title("Audio Waveform (Playing...)")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude")
            self.ax.plot(self.plot_x_data, self.plot_y_data, color='lightgrey', lw=1)
            if self.plot_x_data is not None and len(self.plot_x_data) > 0:
                self.ax.set_xlim(0, self.plot_x_data[-1])
            if self.plot_y_data is not None and len(self.plot_y_data) > 0:
                 self.ax.set_ylim(np.min(self.plot_y_data) * 1.1, np.max(self.plot_y_data) * 1.1)
            self.dynamic_line, = self.ax.plot([], [], color='#1f77b4', lw=1.5, animated=True)
        elif is_live_freq:
            self.ax.grid(True)
            if self.active_plot_func == self.plot_spectrum:
                self.ax.set_title("Live Frequency Spectrum")
                self.ax.set_xlabel("Frequency (Hz)"), self.ax.set_ylabel("Amplitude")
                self.ax.set_xlim(0, self.sample_rate / 2)
                self.ax.set_ylim(0, self.plot_cache.get('spectrum_ymax', 1))
                xf_chunk = np.fft.fftfreq(self.n_fft, 1.0/self.sample_rate)[:self.n_fft//2]
                self.dynamic_line, = self.ax.plot(xf_chunk, np.zeros(self.n_fft//2), lw=1, animated=True)
            else: # PSD
                self.ax.set_title("Live Power Spectral Density")
                self.ax.set_xlabel("Frequency (Hz)"), self.ax.set_ylabel("Power/Frequency (dB/Hz)")
                freqs = np.fft.fftfreq(self.n_fft, 1/self.sample_rate)[:self.n_fft//2]
                self.ax.set_xlim(0, self.sample_rate/2)
                self.ax.set_ylim(self.plot_cache.get('psd_ymin', -100) - 10, self.plot_cache.get('psd_ymax', 0) + 10)
                self.dynamic_line, = self.ax.plot(freqs, np.zeros_like(freqs), lw=1, animated=True)
        else:
            self.active_plot_func() 
            if self.ax.images:
                title = self.ax.get_title().replace(" (Playing...)", "")
                self.ax.set_title(title + " (Playing...)")
                img = self.ax.images[0]
                img.set_animated(True)
                h = abs(self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
                clip_rect = plt.Rectangle((0, self.ax.get_ylim()[0]), 0, h, transform=self.ax.transData)
                img.set_clip_path(clip_rect)
                self.plot_cache['clip_rect'] = clip_rect
        
        self.canvas.draw()
        self.plot_cache['bg'] = self.canvas.copy_from_bbox(self.ax.bbox)

    def _update_plot_animation(self):
        if not self.is_playing:
            return

        # Restore the cached background
        if 'bg' in self.plot_cache:
            self.canvas.restore_region(self.plot_cache['bg'])

        is_live_freq = self.active_plot_func in [self.plot_spectrum, self.plot_psd]
        artist_to_draw = None
        
        if self.active_plot_func == self.plot_waveform:
            if self.dynamic_line is not None:
                self.dynamic_line.set_data(self.plot_x_data[:self.current_frame], self.plot_y_data[:self.current_frame])
                artist_to_draw = self.dynamic_line
        elif is_live_freq:
            start = self.current_frame - self.n_fft
            end = self.current_frame
            if start >= 0 and self.dynamic_line is not None:
                chunk_np = self.audio_data[start:end]
                
                if cuda_available and torch:
                    # --- GPU ACCELERATION with PyTorch ---
                    chunk = torch.from_numpy(chunk_np).to(device, dtype=torch.float32)
                    
                    if self.active_plot_func == self.plot_spectrum:
                        yf = torch.fft.fft(chunk)
                        y_data = 2.0/self.n_fft * torch.abs(yf[0:self.n_fft//2])
                    else: # PSD
                        window = torch.hann_window(self.n_fft, device=device)
                        chunk = chunk * window
                        yf = torch.fft.fft(chunk)
                        psd_data = (torch.abs(yf)**2) / (self.sample_rate * torch.sum(window**2))
                        psd_db = 10 * torch.log10(psd_data[:self.n_fft//2])
                    
                    y_data_cpu = (y_data if self.active_plot_func == self.plot_spectrum else psd_db).cpu().numpy()
                    y_data_cpu[np.isneginf(y_data_cpu)] = -200 
                    self.dynamic_line.set_ydata(y_data_cpu)
                else:
                    # --- CPU fallback with NumPy ---
                    if self.active_plot_func == self.plot_spectrum:
                        yf = np.fft.fft(chunk_np)
                        y_data = 2.0/self.n_fft * np.abs(yf[0:self.n_fft//2])
                        self.dynamic_line.set_ydata(y_data)
                    else: # PSD
                        window = np.hanning(self.n_fft)
                        chunk = chunk_np * window
                        yf = np.fft.fft(chunk)
                        psd_data = (np.abs(yf)**2) / (self.sample_rate * np.sum(window**2))
                        psd_db = 10 * np.log10(psd_data[:self.n_fft//2])
                        psd_db[np.isneginf(psd_db)] = -200
                        self.dynamic_line.set_ydata(psd_db)
                artist_to_draw = self.dynamic_line

        elif 'clip_rect' in self.plot_cache and self.ax.images:
            current_time = self.current_frame / self.sample_rate
            self.plot_cache['clip_rect'].set_width(current_time)
            artist_to_draw = self.ax.images[0]

        # Draw and blit the updated artist
        if artist_to_draw:
            self.ax.draw_artist(artist_to_draw)
            self.canvas.blit(self.ax.bbox)
        
        self.canvas.flush_events()
        
        if self.is_playing:
            self.animation_job = self.master.after(20, self._update_plot_animation) # Faster update rate

    def on_closing(self):
        self._internal_stop_playback_only()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()

