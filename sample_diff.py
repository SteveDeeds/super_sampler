import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import stft, istft

def load_wav(filename):
    """ Load a WAV file and return the sample rate and data. """
    rate, data = wav.read(filename)
    data = data.astype(np.float32)
    #normalize
    data = data/np.max(np.abs(data))
    return rate, data

def save_wav(filename, rate, data):
    """ Save the given data as a WAV file with the specified sample rate. """
    wav.write(filename, rate, data)

def compute_stft(data, fs, nperseg, hop_length):
    """Compute the Short-Time Fourier Transform (STFT) of the input data."""
    noverlap = nperseg - hop_length
    f, t, Zxx = stft(data, fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    return f, t, Zxx

def stft_to_wave(stft_data, fs, nperseg, hop_length):
    """Convert an STFT back to a time-domain signal using inverse STFT."""
    noverlap = nperseg - hop_length
    _, x_rec = istft(stft_data, fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    return x_rec

def combine(spec1, spec2, amount1, amount2, rate, nperseg, hop_length):
    # Compute the difference between the two spectrograms
    mag1, phase1 = np.abs(spec1), np.angle(spec1)
    mag2 = np.abs(spec2)
    mag_diff = mag1*amount1 + mag2*amount2
    mag_diff[mag_diff < 0.0] = 0.0
    diff_Zxx = mag_diff * np.exp(1j * phase1)
    output_waveform = stft_to_wave(diff_Zxx, rate, nperseg, hop_length)
    output_waveform = np.int16(output_waveform / np.max(np.abs(output_waveform)) * 32767)
    save_wav(F"{amount1:0.3f}_{amount2:0.3f}.wav", rate, output_waveform)

def main():
    # Parameters for the spectrogram
    nperseg = 4096  # Number of samples per segment
    hop_length = 256  # Hop length (shift between windows)
    
    # Load both WAV files
    rate1, data1 = load_wav("EP_C4_soft.wav")
    rate2, data2 = load_wav("EP_C4_hard.wav")
    
    # Ensure the files have the same rate and length for comparison
    assert rate1 == rate2, "Sample rates must be the same."
    min_length = min(len(data1), len(data2))
    data1 = data1[:min_length]
    data2 = data2[:min_length]
    
    # Compute spectrograms for both files
    _, _, spec1 = compute_stft(data1, rate1, nperseg, hop_length)
    _, _, spec2 = compute_stft(data2, rate2, nperseg, hop_length)

    min_time_bins = min(spec1.shape[1], spec2.shape[1])

    spec1 = spec1[:, :min_time_bins]
    spec2 = spec2[:, :min_time_bins]

    for i in [-10,-1,-0.1,-0.01,0,0.1,1]:
        combine(spec1, spec2, i, 1.0, rate1, nperseg, hop_length)

if __name__ == "__main__":
    main()
