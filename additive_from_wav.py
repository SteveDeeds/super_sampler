import scipy.io.wavfile as wavfile
from scipy.io.wavfile import write
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from SliceInfo import analyze_frequency
from scipy.ndimage import maximum_filter1d
import os
import csv

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
    harmonic_ranges = [((i - 0.5) * fundamental, (i + 0.5) * fundamental) for i in range(1, 17)]
    
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

def compute_envelope(signal, sr, window_size=70, plot=False):
    """
    Computes the amplitude envelope of a signal using a sliding window maximum filter.
    
    Parameters:
    - signal (numpy array): The input signal
    - sr (int): Sample rate
    - window_size (int): Window size in milliseconds (default is 70)
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

    envelope = envelope[window_size_samples:]
    
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

def calculate_decay_times(envelope, sr, decay_duration=5, plot=False):
    """
    Calculate both the attack and decay times of an envelope, and optionally plot the envelope and synthetic attack-decay envelope.
    
    Parameters:
    - envelope (numpy array): The amplitude envelope of the signal
    - sr (int): Sample rate (in Hz)
    - decay_duration (float): The total duration of the decay phase in seconds (default is 5 seconds)
    - plot (bool): Whether to plot the input and generated envelopes (default is False)
    
    Returns:
    - attack_time (float): The attack time in milliseconds
    - decay_time (float): The decay time as a fraction of the decay duration (in seconds)
    """
    
    amplitude = np.max(np.abs(envelope))
    
    # Normalize the decay portion to 1
    envelope_normalized = envelope / amplitude
    
    # Integrate the area under the normalized decay envelope using the trapezoidal rule
    area_under_curve = np.trapz(envelope_normalized, dx=1 / sr)  # Area under the decay curve
    
    # Calculate decay time: the fraction of the total decay duration (5 seconds) based on the integrated area
    decay_time_ms = area_under_curve * decay_duration  * 1000# Fraction of the 5-second decay
    
    
    # Plot the input envelope and the synthetic envelope if requested
    if plot:
        # Use envelope_generator to create a synthetic envelope based on attack and decay times
        synthetic_envelope = envelope_generator(0, decay_time_ms, sr, plot=False) * amplitude  # decay_time in ms

        plt.figure(figsize=(12, 6))
        
        # Plot the input envelope
        plt.plot(np.linspace(0, len(envelope) / sr, len(envelope)), envelope, label='Input Envelope', color='blue')
        
        # Plot the synthetic envelope
        plt.plot(np.linspace(0, len(synthetic_envelope) / sr, len(synthetic_envelope)), synthetic_envelope, label='Synthetic Envelope', color='orange', linestyle='--')
        
        # Set logarithmic scale for the y-axis
        # plt.yscale('log')  # Logarithmic scale
        
        plt.title('Input Envelope vs Synthetic Envelope')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (log scale)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return decay_time_ms

def generate_composite_wave(amplitudes, frequencies, decay_times, sr=44100, filename='composite_wave.wav', plot=False):
    """
    Generates a composite wave based on provided amplitudes, frequencies, and decay times, then saves to disk.
    
    Parameters:
    - amplitudes (list): List of amplitudes for each harmonic
    - frequencies (list): List of frequencies (fundamental and harmonics)
    - decay_times (list): List of decay times for each harmonic
    - sr (int): Sample rate (default is 44100)
    - filename (str): Output file name for saving the composite wave (default is 'composite_wave.wav')
    - plot (bool): Whether to plot the composite wave (default is False)
    
    Returns:
    - None
    """
    # Determine the maximum decay time in milliseconds and convert to seconds
    max_decay_time_ms = max(decay_times)
    duration = max_decay_time_ms / 1000  # Convert to seconds
    
    # Create a time vector for the given duration
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Initialize the composite signal
    composite_signal = np.zeros_like(t)
    
    # Generate each harmonic and add to the composite signal
    for amplitude, frequency, decay_time in zip(amplitudes, frequencies, decay_times):
        # Generate the sine wave for the harmonic
        harmonic_wave = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Generate the envelope for the harmonic (using the envelope_generator)
        envelope = envelope_generator(attack_time=0, decay_time=decay_time, sr=sr, plot=False)
        
        # Pad the envelope with zeros if it's shorter than the time vector
        envelope_padded = np.pad(envelope, (0, len(t) - len(envelope)), 'constant')
        
        # Apply the envelope to the harmonic wave
        harmonic_wave *= envelope_padded
        
        # Add the harmonic wave to the composite signal
        composite_signal += harmonic_wave
    
    # Normalize the composite signal to the range [-1, 1]
    composite_signal /= np.max(np.abs(composite_signal))
    
    # Convert to 16-bit PCM format (required for saving as WAV)
    composite_signal_int16 = np.int16(composite_signal * 32767)
    
    # Save the composite signal to a WAV file
    write(filename, sr, composite_signal_int16)
    print(f"Composite wave saved as {filename}")
    
    # Plot the composite wave if requested
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(t, composite_signal, label='Composite Wave')
        plt.title('Composite Wave')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()

def process_wave_files_in_directory(directory, output_file):
    # Open the output file for writing
    with open(output_file, 'w', newline='') as csvfile:
        # Create a CSV writer object with tab as the delimiter
        writer = csv.writer(csvfile, delimiter='\t')
        
        # Write the header row (column names)
        header = ['Filename', 'Harmonic Frequencies', 'Amplitudes', 'Decay Times']
        writer.writerow(header)
        
        # Iterate through all the files in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                print(f"Processing file: {filename}")
                
                # Load the WAV file
                filepath = os.path.join(directory, filename)
                sr, y = wavfile.read(filepath)
                y = y / np.max(np.abs(y))  # Normalize to 1
                
                # Analyze the fundamental frequency
                fundamental = analyze_frequency(y, sr)
                compute_envelope(y, sr, plot=False)
                
                # Analyze harmonics
                harmonic_frequencies = analyze_harmonics(y, sr, fundamental, plot=False)

                decay_times = []
                amplitudes = []
                normalized_freq = []
                print(f"Fundamental frequency = {harmonic_frequencies[0]:0.1f}")
                
                # Process each harmonic frequency
                for freq in harmonic_frequencies:
                    # Filter the signal around the harmonic frequency
                    filtered_signal = narrowband_filter(y, sr, freq)
                    normalized_freq.append(freq/harmonic_frequencies[0])

                    # Compute the envelope
                    envelope = compute_envelope(filtered_signal, sr, plot=False)

                    amplitude = np.max(np.abs(filtered_signal))
                    
                    # Find the decay time
                    decay_time_ms = calculate_decay_times(envelope, sr, plot=False)
                    print(f"For {normalized_freq[-1]:0.4f}, Amplitude = {amplitude:0.3f}, Decay time = {decay_time_ms:0.2f} ms")
                    decay_times.append(decay_time_ms)
                    amplitudes.append(amplitude)

                # Save the results to the TSV file
                # Write the filename, harmonic frequencies, amplitudes, and decay times without quotes
                row = [filename] + normalized_freq + amplitudes + decay_times
                writer.writerow(row)


            # # Generate and save the composite wave
            # output_filename = f"composite_{filename}"
            # generate_composite_wave(amplitudes=amplitudes, frequencies=harmonic_frequencies, decay_times=decay_times, filename=output_filename)
            # print(f"Processed {filename} and saved as {output_filename}")

directory = '.\\output\\samples'  # Replace with the path to your directory containing WAV files
process_wave_files_in_directory(directory, "output.tsv")
