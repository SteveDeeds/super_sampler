import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d

def load_wav_file(filename):
    """Load a WAV file and normalize it."""
    rate, data = wavfile.read(filename)
    if data.ndim > 1:  # Convert to mono if stereo
        data = data.mean(axis=1)
    return rate, data

def calculate_envelope(data, window_size):
    """Calculate the envelope using a maximum filter."""
    envelope = maximum_filter1d(np.abs(data), size=window_size)
    return envelope

def plot_envelopes(time, envelope1, envelope2, label1, label2):
    """Plot two envelopes superimposed."""
    plt.figure(figsize=(10, 6))
    plt.fill_between(time, envelope1, color="red", alpha=0.5, label=label1, linewidth=0)
    plt.fill_between(time, envelope2, color="blue", alpha=0.5, label=label2, linewidth=0)
    plt.title("Waveform Envelopes")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    #plt.show()

def plot_spectrograms(data1, data2, rate, nfft=4096, noverlap=2048, file1="", file2=""):
    """Plot spectrograms of two wave files using red and blue channels."""
    # Compute spectrograms
    Pxx1, freqs1, bins1 = plt.mlab.specgram(data1, NFFT=nfft, Fs=rate, noverlap=noverlap)
    Pxx2, freqs2, bins2 = plt.mlab.specgram(data2, NFFT=nfft, Fs=rate, noverlap=noverlap)

    # Normalize for visualization
    Pxx1 = 10 * np.log10(Pxx1 + 1e-10)
    Pxx2 = 10 * np.log10(Pxx2 + 1e-10)
    Pxx1 = (Pxx1 - Pxx1.min()) / (Pxx1.max() - Pxx1.min())
    Pxx2 = (Pxx2 - Pxx2.min()) / (Pxx2.max() - Pxx2.min())

    # Create RGB image with red and blue channels
    rgb_image = np.zeros((Pxx1.shape[0], Pxx1.shape[1], 3))
    rgb_image[:, :, 0] = Pxx2/2  # Red channel
    rgb_image[:, :, 1] = np.abs(Pxx2-Pxx1)  # Green channel
    rgb_image[:, :, 2] = Pxx1/2  # Blue channel

    plt.figure(figsize=(10, 6))
    plt.imshow(
        rgb_image,
        extent=[bins1[0], bins1[-1], freqs1[0], freqs1[-1]],
        origin='lower',
        aspect='auto'
    )
    plt.title(f"Spectrograms of {file1} (Blue) and {file2} (Red)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    #plt.show()

def main():
    # File paths for the WAV files
    file1 = "CP33-EPiano1.wav"
    file2 = "CP33-EPiano1-comp.wav"

    # Load WAV files
    rate1, data1 = load_wav_file(file1)
    rate2, data2 = load_wav_file(file2)

    # Ensure both signals have the same sample rate
    if rate1 != rate2:
        raise ValueError("Sample rates of the two WAV files do not match.")

    # Match the lengths of the two signals
    min_length = min(len(data1), len(data2))
    data1 = data1[:min_length]
    data2 = data2[:min_length]

    # Calculate the envelope for each signal
    window_size = int(rate1 * 0.03)  # 30 ms window
    envelope1 = calculate_envelope(data1, window_size)
    envelope2 = calculate_envelope(data2, window_size)

    # Time axis
    time = np.linspace(0, len(data1) / rate1, num=len(data1))

    # Plot the envelopes
    plot_envelopes(time, envelope1, envelope2, file1, file2)

    # Plot the spectrograms
    plot_spectrograms(data1, data2, rate1, file1=file1, file2=file2)

    plt.show()

if __name__ == "__main__":
    main()
