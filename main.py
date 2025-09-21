import tkinter as tk
from tkinter import filedialog, messagebox
import wave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display

class AudioAnalyzerApp:
    """
    A Python application to analyze and visualize audio files.
    """
    def __init__(self, master):
        """
        Initializes the Audio Analyzer application.

        Args:
            master: The root tkinter window.
        """
        self.master = master
        self.master.title("Audio Analyzer")
        self.master.geometry("950x700")

        self.filepath = None
        self.audio_data = None
        self.sample_rate = None

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

        # Analysis buttons - Row 1
        self.analysis_frame1 = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.analysis_frame1.pack(pady=2)
        self.waveform_button = tk.Button(self.analysis_frame1, text="Waveform", command=self.plot_waveform, state=tk.DISABLED, font=("Helvetica", 12))
        self.waveform_button.pack(side=tk.LEFT, padx=5)
        self.spectrum_button = tk.Button(self.analysis_frame1, text="Frequency Spectrum", command=self.plot_spectrum, state=tk.DISABLED, font=("Helvetica", 12))
        self.spectrum_button.pack(side=tk.LEFT, padx=5)
        self.psd_button = tk.Button(self.analysis_frame1, text="Power Spectral Density", command=self.plot_psd, state=tk.DISABLED, font=("Helvetica", 12))
        self.psd_button.pack(side=tk.LEFT, padx=5)

        # Analysis buttons - Row 2
        self.analysis_frame2 = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.analysis_frame2.pack(pady=2)
        self.spectrogram_button = tk.Button(self.analysis_frame2, text="Spectrogram", command=self.plot_spectrogram, state=tk.DISABLED, font=("Helvetica", 12))
        self.spectrogram_button.pack(side=tk.LEFT, padx=5)
        self.chromagram_button = tk.Button(self.analysis_frame2, text="Chromagram", command=self.plot_chromagram, state=tk.DISABLED, font=("Helvetica", 12))
        self.chromagram_button.pack(side=tk.LEFT, padx=5)
        self.tempogram_button = tk.Button(self.analysis_frame2, text="Tempogram", command=self.plot_tempogram, state=tk.DISABLED, font=("Helvetica", 12))
        self.tempogram_button.pack(side=tk.LEFT, padx=5)
        self.tonnetz_button = tk.Button(self.analysis_frame2, text="Tonnetz", command=self.plot_tonnetz, state=tk.DISABLED, font=("Helvetica", 12))
        self.tonnetz_button.pack(side=tk.LEFT, padx=5)


        # Plotting canvas
        self.plot_frame = tk.Frame(self.main_frame, bg="white", relief=tk.SUNKEN, borderwidth=1)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.fig = plt.Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def select_file(self):
        """
        Opens a file dialog to select an audio file.
        """
        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(("Audio Files", "*.wav *.mp3 *.flac"), ("All files", "*.*"))
        )
        if filepath:
            try:
                self.filepath = filepath
                self.file_label.config(text=self.filepath.split('/')[-1])
                self.load_audio()
                self.waveform_button.config(state=tk.NORMAL)
                self.spectrum_button.config(state=tk.NORMAL)
                self.psd_button.config(state=tk.NORMAL)
                self.spectrogram_button.config(state=tk.NORMAL)
                self.chromagram_button.config(state=tk.NORMAL)
                self.tempogram_button.config(state=tk.NORMAL)
                self.tonnetz_button.config(state=tk.NORMAL)
                self.plot_waveform() # Plot waveform by default
            except Exception as e:
                messagebox.showerror("Error", f"Could not open or read the audio file.\n{e}")
                self.reset()

    def load_audio(self):
        """
        Loads the audio data from the selected file using librosa.
        """
        if self.filepath:
            self.audio_data, self.sample_rate = librosa.load(self.filepath, sr=None, mono=True)

    def reset(self):
        """
        Resets the application to its initial state.
        """
        self.filepath = None
        self.audio_data = None
        self.sample_rate = None
        self.file_label.config(text="No file selected")
        self.waveform_button.config(state=tk.DISABLED)
        self.spectrum_button.config(state=tk.DISABLED)
        self.psd_button.config(state=tk.DISABLED)
        self.spectrogram_button.config(state=tk.DISABLED)
        self.chromagram_button.config(state=tk.DISABLED)
        self.tempogram_button.config(state=tk.DISABLED)
        self.tonnetz_button.config(state=tk.DISABLED)
        self.clear_plot()

    def clear_plot(self):
        """
        Clears the plot figure completely.
        """
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.canvas.draw()

    def plot_waveform(self):
        """
        Plots the audio waveform (Amplitude vs. Time).
        """
        if self.audio_data is not None:
            self.clear_plot()
            librosa.display.waveshow(y=self.audio_data, sr=self.sample_rate, ax=self.ax)
            self.ax.set_title("Audio Waveform")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude")
            self.canvas.draw()

    def plot_spectrum(self):
        """
        Plots the frequency spectrum (Amplitude vs. Frequency) using FFT.
        """
        if self.audio_data is not None:
            self.clear_plot()
            n = len(self.audio_data)
            T = 1.0 / self.sample_rate
            yf = np.fft.fft(self.audio_data)
            xf = np.fft.fftfreq(n, T)[:n//2]
            self.ax.plot(xf, 2.0/n * np.abs(yf[0:n//2]))
            self.ax.set_title("Frequency Spectrum")
            self.ax.set_xlabel("Frequency (Hz)")
            self.ax.set_ylabel("Amplitude")
            self.ax.grid()
            self.canvas.draw()

    def plot_psd(self):
        """
        Plots the Power Spectral Density of the audio signal.
        """
        if self.audio_data is not None:
            self.clear_plot()
            self.ax.psd(self.audio_data, Fs=self.sample_rate, NFFT=2048)
            self.ax.set_title("Power Spectral Density")
            self.ax.set_xlabel("Frequency (Hz)")
            self.ax.set_ylabel("Power/Frequency (dB/Hz)")
            self.canvas.draw()
            
    def plot_spectrogram(self):
        """
        Plots a spectrogram of the audio signal.
        """
        if self.audio_data is not None:
            self.clear_plot()
            D = librosa.stft(self.audio_data)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(S_db, sr=self.sample_rate, x_axis='time', y_axis='log', ax=self.ax)
            self.fig.colorbar(img, ax=self.ax, format='%+2.0f dB', label='Intensity')
            self.ax.set_title('Spectrogram')
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Frequency (Hz)")
            self.canvas.draw()

    def plot_chromagram(self):
        """
        Plots a chromagram of the audio signal.
        """
        if self.audio_data is not None:
            self.clear_plot()
            chroma = librosa.feature.chroma_stft(y=self.audio_data, sr=self.sample_rate)
            img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=self.ax, sr=self.sample_rate)
            self.fig.colorbar(img, ax=self.ax, label='Intensity')
            self.ax.set_title('Chromagram')
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Pitch Class")
            self.canvas.draw()

    def plot_tempogram(self):
        """
        Plots a tempogram to visualize tempo changes over time.
        """
        if self.audio_data is not None:
            self.clear_plot()
            onset_env = librosa.onset.onset_strength(y=self.audio_data, sr=self.sample_rate)
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sample_rate)
            img = librosa.display.specshow(tempogram, sr=self.sample_rate, x_axis='time', y_axis='tempo', ax=self.ax, cmap='magma')
            self.fig.colorbar(img, ax=self.ax, label='Strength')
            self.ax.set_title('Tempogram')
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Tempo (BPM)")
            self.canvas.draw()

    def plot_tonnetz(self):
        """
        Plots the Tonal Centroid Network (Tonnetz) of the audio.
        """
        if self.audio_data is not None:
            self.clear_plot()
            tonnetz = librosa.feature.tonnetz(y=self.audio_data, sr=self.sample_rate)
            img = librosa.display.specshow(tonnetz, sr=self.sample_rate, x_axis='time', y_axis='tonnetz', ax=self.ax, cmap='coolwarm')
            self.fig.colorbar(img, ax=self.ax)
            self.ax.set_title('Tonal Centroid Network (Tonnetz)')
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Tonal Centroids")
            self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()

