import scipy.io.wavfile as wavfile
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from SliceInfo import analyze_frequency
from scipy.ndimage import maximum_filter1d

def envelope_generator(attack_time, decay_time, sr=44100, plot=False):
    """
    Generates an attack-decay envelope with exponential decay.
    
    Parameters:
    - attack_time (ms): Time for the attack phase
    - decay_time (ms): Time for the decay phase
    - sr (int): Sample rate (default is 44100)
    - plot (bool): Whether to plot the envelope (default is False)
    
    Returns:
    - envelope (numpy array): The generated envelope
    """
    
    # Convert milliseconds to samples
    attack_samples = int(sr * (attack_time / 1000))
    decay_samples = int(sr * (decay_time / 1000))
    total_samples = attack_samples + decay_samples
    
    # Generate the attack phase (linear rise)
    attack_phase = np.linspace(0, 1, attack_samples)
    
    # Generate the decay phase (exponential decay)
    decay_phase = np.exp(-np.linspace(0, 5, decay_samples))
    
    # Combine the phases
    envelope = np.concatenate((attack_phase, decay_phase))
    
    # Plot the envelope if requested
    if plot:
        plt.plot(envelope)
        plt.title('Attack-Decay Envelope')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.show()
    
    return envelope

def find_strongest_in_range(freqs, mags, range_min, range_max):
    indices = np.where((freqs >= range_min) & (freqs <= range_max))[0]
    if len(indices) == 0:
        return None
    max_index = indices[np.argmax(mags[indices])]
    return freqs[max_index]

def analyze_harmonics(y, sr, fundamental, plot=False):
    # Perform FFT
    Y = fft(y)
    magnitude = np.abs(Y)
    frequencies = np.fft.fftfreq(len(magnitude), 1/sr)
    
    # Identify harmonic ranges
    harmonic_ranges = [((i - 0.5) * fundamental, (i + 0.5) * fundamental) for i in range(1, 9)]
    
    # Find the strongest frequency near each harmonic
    harmonic_frequencies = []
    for h_range in harmonic_ranges:
        print(f"Harmonic range: {h_range}")
        strongest_freq = find_strongest_in_range(frequencies, magnitude, h_range[0], h_range[1])
        if strongest_freq is not None:
            harmonic_frequencies.append(strongest_freq)
    
    # Create an array of perfect harmonics
    perfect_harmonics = [fundamental * (i + 1) for i in range(8)]
    
    print("Fundamental:", fundamental)
    print("Harmonic Frequencies:", harmonic_frequencies)
    print("Perfect Harmonics:", perfect_harmonics)
    if plot:
        # Plot the magnitude spectrum
        plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
        plt.title('Magnitude Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.show()
    
    return harmonic_frequencies

def narrowband_filter(signal, sr, target_freq):
    """
    Filters the signal by extracting the bin closest to the target frequency using STFT and ISTFT.
    
    Parameters:
    - signal (numpy array): The input signal
    - sr (int): Sample rate
    - target_freq (float): Target frequency to isolate
    
    Returns:
    - filtered_signal (numpy array): The filtered signal
    """
    
    # Compute the STFT
    f, t, Zxx = stft(signal, sr)
    
    # Find the index of the closest frequency bin to the target frequency
    closest_bin = np.argmin(np.abs(f - target_freq))
    
    # Create a mask to isolate the desired frequency bin
    mask = np.zeros_like(Zxx)
    mask[closest_bin, :] = Zxx[closest_bin, :]
    
    # Reconstruct the time-domain signal using the ISTFT
    _, filtered_signal = istft(mask, sr)
    
    return filtered_signal

def compute_envelope(signal, sr, window_size=100, plot=False):
    """
    Computes the amplitude envelope of a signal using a sliding window maximum filter.
    
    Parameters:
    - signal (numpy array): The input signal
    - sr (int): Sample rate
    - window_size (int): Window size in milliseconds (default is 100)
    - plot (bool): Whether to plot the signal and envelope (default is False)
    
    Returns:
    - amplitude_envelope (numpy array): The computed amplitude envelope
    """
    
    # Convert window size from milliseconds to samples
    window_size_samples = window_size * sr // 1000
    
    # Pad the signal with zeros by half the window size
    padded_signal = np.pad(signal, (window_size_samples // 2,), 'constant')
    
    # Compute the envelope using a sliding window maximum filter
    envelope = maximum_filter1d(padded_signal, size=window_size_samples)
    
    # Create a time vector for plotting
    time_vector = np.linspace(0, len(signal) / sr, len(signal))
    
    # Plot the signal and envelope if requested
    if plot:
        plt.figure(figsize=(12, 6))
        
        plt.plot(time_vector, signal, label='Original Signal')
        plt.plot(time_vector, envelope[:len(signal)], label='Amplitude Envelope', color='orange')
        plt.title('Signal and Amplitude Envelope')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()
    
    return envelope[:len(signal)]

def calculate_attack_and_decay_times(envelope, sr, decay_duration=5):
    """
    Calculate both the attack and decay times of an envelope.
    
    Parameters:
    - envelope (numpy array): The amplitude envelope of the signal
    - sr (int): Sample rate (in Hz)
    - threshold (float): The amplitude level to define the end of the decay (default is 1% of the maximum amplitude)
    - decay_duration (float): The total duration of the decay phase in seconds (default is 5 seconds)
    
    Returns:
    - attack_time (float): The attack time in milliseconds
    - decay_time (float): The decay time as a fraction of the decay duration (in seconds)
    """
    
    # 1. Calculate attack time: Find the point where the envelope reaches the maximum value
    max_value = np.max(envelope)
    attack_end_index = np.argmax(envelope >= max_value)  # Find where it reaches the peak
    attack_time_samples = attack_end_index
    attack_time_ms = (attack_time_samples / sr) * 1000  # Convert to milliseconds
    
    # 2. Calculate decay time:
    # Find the portion of the envelope after the attack phase
    decay_start_index = attack_end_index
    decay_envelope = envelope[decay_start_index:]  # Get the decay portion of the envelope
    
    # Normalize the decay portion to 1
    decay_envelope_normalized = decay_envelope / np.max(decay_envelope)
    
    # Integrate the area under the normalized decay envelope using the trapezoidal rule
    area_under_curve = np.trapz(decay_envelope_normalized, dx=1 / sr)  # Area under the decay curve
    
    # Calculate decay time: the fraction of the total decay duration (5 seconds) based on the integrated area
    decay_time_ms = area_under_curve / decay_duration *1000  # Fraction of the 5-second decay
    
    return attack_time_ms, decay_time_ms

# Load the WAV file
filename = 'CP33-EP_C4.wav'
sr, y = wavfile.read(filename)
fundamental = analyze_frequency(y, sr)

compute_envelope(y,sr,plot=True)

# Analyze harmonics
harmonic_frequencies = analyze_harmonics(y, sr, fundamental, plot=False)

# Process each harmonic frequency
for freq in harmonic_frequencies:
    # Filter the signal around the harmonic frequency
    filtered_signal = narrowband_filter(y, sr, freq)
    
    # Compute the envelope
    envelope = compute_envelope(filtered_signal, sr, plot=True)
    
    # Find the attack time
    attack_time_ms, decay_time_ms = calculate_attack_and_decay_times(envelope, sr)
    print(f"For {freq}Hz, Attack time = {attack_time_ms:.2f} ms Decay time = {decay_time_ms:.2f} ms")
