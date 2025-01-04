import os
import numpy as np
from scipy.io.wavfile import read
from scipy.fft import fftfreq, fft
from scipy.signal import resample
from scipy.signal import stft, istft

class SliceInfo:
    """
    Represents information about a slice of a .wav file.
    """
    def __init__(self, index, frequency_hz, musical_note, peak_amplitude, velocity, slice_data):
        self.index = index
        self.frequency_hz = frequency_hz
        self.musical_note = musical_note
        self.peak_amplitude = peak_amplitude
        self.velocity = velocity
        self.slice_data = slice_data

    def __repr__(self):
        return (f"Slice {self.index}: {self.frequency_hz:.2f} Hz ({self.musical_note}), "
                f"Peak Amplitude: {self.peak_amplitude}, Velocity: {self.velocity}")
    
def frequency_to_note(freq):
    if freq <= 0:
        return "Unknown", None

    A4 = 440.0
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Calculate the MIDI note number
    note_number = 12 * np.log2(freq / A4) + 69
    note_index = int(round(note_number)) % 12
    octave = (int(round(note_number)) // 12) - 1
    
    # Return both the note name and the note number (MIDI number)
    note_name = f"{notes[note_index]}{octave}"
    
    return note_name, int(round(note_number))

def analyze_frequency(wave_data, sample_rate):
    N = len(wave_data)
    tone = wave_data[N//4:N//2]
    yf = np.abs(fft(tone))
    xf = fftfreq(len(tone), 1 / sample_rate)
    idx = np.argmax(np.abs(yf))
    # check if it's the 2nd harmonic that we are seeing as the strongest.
    if yf[idx//2] > yf[idx]/10:
        f = xf[idx//2]
    else:
        f = xf[idx]
    return f

# def analyze_frequency(signal, sample_rate):
#     N = len(signal)
#     tone = signal[N//8:N//2]
#     N = len(tone)
#     spectrum = np.abs(np.fft.fft(tone, n=N))[:N // 2]
#     freqs = np.fft.fftfreq(n=N, d=1.0/sample_rate)[:N // 2]

#     # # weight the lower frequencies as more likely
#     # scale = 1.0/(np.arange(len(spectrum)) + 1.0)
#     # spectrum = spectrum * scale

#     # # Find the frequencies of the top 5 peaks
#     # frequencies = np.fft.fftfreq(N, 1 / sample_rate)[:N // 2]
#     # top_indices = np.argsort(spectrum)[-20:][::-1]  # Indices of the largest peaks
#     # top_frequencies = frequencies[top_indices]
#     # print(F"median={np.median(top_frequencies)} min={np.min(top_frequencies)} ")

#     # Downsample harmonics and ensure equal lengths
#     harmonics = [spectrum[::i] for i in range(1, 3)]
#     min_length = min(len(h) for h in harmonics)
#     harmonics = [h[:min_length] for h in harmonics]  # Trim all harmonics to the shortest length

#     # Harmonic Product Spectrum calculation
#     hps = np.sum(harmonics, axis=0)
#     freqs = freqs[:len(hps)]
#     spectrum = spectrum[:len(hps)]
#     peak_index = np.argmax(hps)

#     # plot
#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.semilogx(freqs, hps, label='Harmonic Product Spectrum', color='blue', linewidth=2)
#     plt.semilogx(freqs, spectrum, label='Spectrum', color='orange', linestyle='--', linewidth=1.5)
#     plt.title('Harmonic Product Spectrum')
#     plt.xlabel('Frequency Index')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.grid(True)
#     plt.waitforbuttonpress()

#     # Convert the peak index to frequency
#     freq = sample_rate * peak_index / N
#     return freq
def calculate_velocity(peak_amplitude, max_amplitude):
    """
    Calculates velocity relative to the loudest slice, scaled to a max of 127.
    """
    if max_amplitude == 0:
        return 0
    return int((peak_amplitude / max_amplitude) * 127)

# def change_pitch(audio_data, sample_rate, target_frequency):
#     """
#     Changes the pitch of the audio data by adjusting the sample rate.

#     Parameters:
#     - audio_data: np.ndarray, the waveform data as a numpy array.
#     - sample_rate: int, the sample rate of the audio.
#     - target_frequency: float, the desired frequency in Hz.

#     Returns:
#     - np.ndarray: The pitch-adjusted audio data.
#     """
#     # Analyze the current dominant frequency
#     current_frequency = analyze_frequency(audio_data, sample_rate)

#     if current_frequency <= 0:
#         raise ValueError("The audio signal contains no detectable frequency.")

#     # Calculate the pitch adjustment factor
#     pitch_factor = current_frequency / target_frequency

#     # Change the pitch by adjusting the sample rate
#     resampled_audio = resample(audio_data, int(len(audio_data) * pitch_factor))

#     # Return the pitch-adjusted audio
#     return resampled_audio.astype(np.float32)

def change_pitch(audio_data, sample_rate, target_frequency):
    """
    Changes the pitch of the audio data without altering the playback duration.

    Parameters:
    - audio_data: np.ndarray, the waveform data as a numpy array.
    - sample_rate: int, the sample rate of the audio.
    - target_frequency: float, the desired frequency in Hz.

    Returns:
    - np.ndarray: The pitch-adjusted audio data.
    """
    # Analyze the current dominant frequency
    current_frequency = analyze_frequency(audio_data, sample_rate)

    if current_frequency <= 0:
        raise ValueError("The audio signal contains no detectable frequency.")

    # Calculate the pitch shift ratio
    pitch_shift_ratio = target_frequency / current_frequency

    # Perform Short-Time Fourier Transform (STFT)
    n_fft = 4096
    hop_length = n_fft // 8
    f, t, Zxx = stft(audio_data, fs=sample_rate, nperseg=n_fft, noverlap=n_fft-hop_length)

    # Modify the frequencies by the pitch shift ratio
    num_bins = Zxx.shape[0]
    stretched_Zxx = np.zeros_like(Zxx, dtype=np.complex64)
    for i in range(num_bins):
        target_bin = int(i / pitch_shift_ratio)
        if 0 <= target_bin < num_bins:
            stretched_Zxx[i] = Zxx[target_bin]

    # Reconstruct the time-domain signal using ISTFT
    _, pitch_shifted_audio = istft(stretched_Zxx, fs=sample_rate, nperseg=n_fft, noverlap=n_fft-hop_length)

    return pitch_shifted_audio.astype(np.float32)

def tune_to_nearest_note(audio_data, sample_rate):
    """
    Tunes an audio sample to the nearest musical note by adjusting its pitch.

    Parameters:
    - audio_data: np.ndarray, the waveform data as a numpy array.
    - sample_rate: int, the sample rate of the audio.

    Returns:
    - tuned_audio_data: np.ndarray, the pitch-corrected audio data.
    - target_note: str, the name of the target musical note.
    """

    # Step 1: Analyze the dominant frequency in the audio
    dominant_freq = analyze_frequency(audio_data, sample_rate)

    if dominant_freq <= 0:
        return audio_data, "Unknown"

    # Step 2: Find the nearest musical note and its frequency
    A4 = 440.0  # Reference frequency for A4
    note_number = 12 * np.log2(dominant_freq / A4) + 49
    target_freq = A4 * 2 ** ((round(note_number) - 49) / 12)

    # Step 3: Adjust pitch to match the nearest note
    tuned_audio_data = change_pitch(audio_data, sample_rate, target_freq)

    return tuned_audio_data

def load_wav(file_path):
    """
    Loads a .wav file, preprocesses it by normalizing and converting to mono if necessary.

    Parameters:
    - file_path: str, the path to the .wav file.

    Returns:
    - sample_rate: int, the sample rate of the .wav file.
    - wave_data: np.ndarray, the preprocessed wave data.
    """
    sample_rate, wave_data = read(file_path)

    # Normalize data if needed
    if wave_data.dtype == np.int16:
        wave_data = (wave_data / 32768.0).astype(np.float32)
    elif wave_data.dtype == np.int32:
        wave_data = (wave_data / 2147483648.0).astype(np.float32)
    elif wave_data.dtype == np.uint8:
        wave_data = ((wave_data - 128) / 128.0).astype(np.float32)

    # Convert stereo to mono if necessary
    if len(wave_data.shape) > 1:
        wave_data = wave_data.mean(axis=1)

    return sample_rate, wave_data

def prepare_directories(output_dir):
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "wavetables"), exist_ok=True)