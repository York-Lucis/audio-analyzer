# Audio Analyzer

A comprehensive desktop application for audio analysis and visualization built with Python and PyQt6. This tool provides both single audio file analysis and dual audio file comparison capabilities with real-time playback and multiple visualization modes.

## Features

### 🎵 Single Audio Analysis
- **Waveform Visualization**: View time-domain representation of audio signals
- **Frequency Spectrum**: Analyze frequency content with FFT-based spectrum analysis
- **Power Spectral Density**: Examine power distribution across frequencies
- **Spectrogram**: Time-frequency representation showing how spectral content changes over time
- **Chromagram**: Visualize harmonic content and chord progressions
- **Tempogram**: Analyze rhythmic patterns and tempo variations
- **Tonnetz**: Display tonal relationships and harmonic networks

### 🔄 Dual Audio Comparison
- Side-by-side comparison of two audio files
- Synchronized visualization across multiple analysis types
- Perfect for A/B testing, audio quality comparison, and forensic analysis

### 🎮 Interactive Features
- **Real-time Playback**: Play audio with synchronized visualization updates
- **Live Animation**: Watch analysis plots update in real-time during playback
- **CUDA Support**: Automatic GPU acceleration when available
- **Multiple Format Support**: WAV, MP3, FLAC audio file formats

## Screenshots

*[Screenshots would be added here showing the application interface]*

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (included in the repository)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/York-Lucis/audio-analyzer.git
   cd audio-analyzer
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

### Dependencies

The application uses the following key libraries:
- **PyQt6**: Modern GUI framework for the desktop interface
- **librosa**: Advanced audio analysis and feature extraction
- **pyqtgraph**: High-performance plotting and visualization
- **sounddevice**: Real-time audio playback
- **pydub**: Audio file processing and format conversion
- **numpy**: Numerical computations and array operations

## Usage

### Single Audio Analysis

1. Launch the application
2. Click "Select Audio File" to load your audio file
3. Choose from available analysis types:
   - **Waveform**: Basic time-domain visualization
   - **Frequency Spectrum**: FFT-based frequency analysis
   - **Power Spectral Density**: Power distribution analysis
   - **Spectrogram**: Time-frequency heatmap
   - **Chromagram**: Harmonic content visualization
   - **Tempogram**: Rhythmic pattern analysis
   - **Tonnetz**: Tonal relationship mapping
4. Use playback controls to play audio with synchronized visualization

### Dual Audio Comparison

1. Select "Dual Audio Comparison" from the side menu
2. Load two audio files using the respective "Select Audio File" buttons
3. Choose your analysis type to compare both files side-by-side
4. Analyze differences in spectral content, harmonic structure, or rhythmic patterns

## Technical Details

### Architecture
- **Modular Design**: Separate widgets for single and dual analysis modes
- **Caching System**: Intelligent plot caching for improved performance
- **Real-time Processing**: Live audio analysis during playback
- **Cross-platform**: Works on Windows, macOS, and Linux

### Performance Optimizations
- **Numba Integration**: Automatic JIT compilation for faster computations
- **CUDA Support**: GPU acceleration for compatible systems
- **Plot Caching**: Reduces redundant calculations
- **Downsampling**: Efficient rendering of large audio files

### Audio Processing
- **Automatic Mono Conversion**: Multi-channel audio is converted to mono for analysis
- **Normalization**: Audio signals are normalized for consistent visualization
- **Window Functions**: Proper windowing for FFT analysis
- **Real-time Streaming**: Low-latency audio playback with visualization sync

## Development

### Project Structure
```
audio-analyzer/
├── main.py              # Main application entry point
├── requirements.txt     # Python dependencies
├── ffmpeg/             # FFmpeg binaries and documentation
│   ├── bin/            # Executable files
│   ├── doc/            # Documentation
│   └── presets/        # Encoding presets
└── README.md           # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings for classes and functions
- Maintain consistent indentation and formatting

## Troubleshooting

### Common Issues

**Audio playback not working:**
- Ensure your system has audio output devices configured
- Check that sounddevice can access your audio system
- On Linux, you may need to install additional audio libraries

**FFmpeg errors:**
- The application includes FFmpeg binaries, but if you encounter issues, ensure FFmpeg is properly installed on your system
- Check file permissions for the ffmpeg directory

**Performance issues:**
- For large audio files, consider using shorter segments for analysis
- Ensure you have sufficient RAM for the audio file size
- CUDA acceleration requires compatible NVIDIA GPU and drivers

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **librosa**: For comprehensive audio analysis capabilities
- **PyQt6**: For the modern GUI framework
- **pyqtgraph**: For high-performance plotting
- **FFmpeg**: For audio processing and format support

## Contact

**York-Lucis**
- GitHub: [@York-Lucis](https://github.com/York-Lucis)

---

*Built with ❤️ for audio analysis and visualization*
