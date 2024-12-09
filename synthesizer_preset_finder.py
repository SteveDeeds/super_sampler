import numpy as np
from scipy.io.wavfile import read as wavfile_read
from scipy.signal import spectrogram
from scipy.signal import istft
from scipy.io.wavfile import write
import matplotlib.pyplot as plt


def convert_wave_to_spectrogram(waveform, sample_rate=44100, n_fft=2048, hop_length=512):
    frequencies, times, S = spectrogram(waveform, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length, mode='magnitude')
    S_db = 10 * np.log10(S + 1e-10)  # Convert to decibels and avoid log(0)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, S_db, shading='auto', cmap='inferno')
    plt.colorbar(label='Amplitude (dB)')
    plt.title('Spectrogram')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, sample_rate // 2)  # Limit to the Nyquist frequency
    plt.show()
    plt.waitforbuttonpress()

    return frequencies, times, S_db

def load_wav_and_convert_to_spectrogram_scipy(file_path, n_fft=2048, hop_length=512):
    """
    Loads a .wav file and converts it to a spectrogram using scipy.
    Also returns the sound duration in seconds.
    """
    sr, audio = wavfile_read(file_path)
    if audio.dtype != np.float32:
        audio = audio / np.max(np.abs(audio), axis=0)  # Normalize to [-1, 1]
    frequencies, times, S_db = convert_wave_to_spectrogram(audio, sample_rate=sr, n_fft=n_fft, hop_length=hop_length)
    duration = len(audio) / sr
    return frequencies, times, S_db, sr, duration


def convert_spectrogram_to_waveform(spectrogram_db, sample_rate=44100, n_fft=2048, hop_length=512):
    spectrogram = 10 ** (spectrogram_db / 10)
    duration = spectrogram.shape[1] * (n_fft - hop_length) / sample_rate
    phase = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))
    complex_spectrogram = spectrogram * phase
    waveform = istft(complex_spectrogram, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length, nfft=n_fft)[1]
    return waveform, duration

# Envelope function (ADSR)
def envelope(attack, decay, sustain, release, duration, sample_rate):
    attack_end = attack
    decay_end = attack + decay
    release_start = duration - release
    t = np.linspace(0.0, duration, int(sample_rate * duration))
    envelope = np.zeros_like(t)

    # Attack phase
    envelope[(t >= 0) & (t < attack_end)] = t[(t >= 0) & (t < attack_end)] / attack
    # Decay phase
    envelope[(t >= attack_end) & (t < decay_end)] = 1 - ((1 - sustain) * (t[(t >= attack_end) & (t < decay_end)] - attack_end) / decay)
    # Sustain phase
    envelope[(t >= decay_end) & (t < release_start)] = sustain
    # Release phase
    envelope[(t >= release_start) & (t <= duration)] = sustain * (1 - (t[(t >= release_start) & (t <= duration)] - release_start) / release)

    # # Plot the envelope
    # plt.figure(figsize=(10, 4))
    # plt.plot(t, envelope, label='ADSR Envelope')
    # plt.title("ADSR Envelope")
    # plt.xlabel("Time (seconds)")
    # plt.ylabel("Amplitude")
    # plt.ylim(-0.1, 1.1)  # Ensure the y-axis is within the valid range
    # plt.grid()
    # plt.legend()
    # plt.show()

    return envelope

def synthesizer(parameters, sample_rate=44100, target_spectrogram=None, duration=None, n_fft=2048, hop_length=512):
    t = np.linspace(0.0, duration, int(sample_rate * duration))

    if target_spectrogram is not None and duration is None:
        duration = target_spectrogram.shape[1] * (n_fft - hop_length) / sample_rate
    frequency, amplitude, attack, decay, sustain, release = parameters
    env = envelope(attack, decay, sustain, release, duration, sample_rate=sample_rate)
    waveform = amplitude * env * np.sin(2 * np.pi * frequency * t)

    # # Plot the waveform
    # plt.figure(figsize=(10, 4))
    # plt.plot(t, waveform, label="Waveform")
    # plt.title("Synthesized Waveform")
    # plt.xlabel("Time (seconds)")
    # plt.ylabel("Amplitude")
    # plt.grid()
    # plt.legend()
    # plt.show()

    return waveform

# Save waveform to disk
def save_waveform_to_disk(waveform, sample_rate=44100, output_path="output.wav"):
    # Ensure the waveform is 1D (mono)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0)  # Convert stereo to mono by averaging
    
    # Normalize waveform to the range of int16
    waveform = np.int16(waveform / np.max(np.abs(waveform)) * 32767)

    # Save the waveform to a .wav file
    write(output_path, sample_rate, waveform)
    print(f"Waveform saved to {output_path}")

def gradient_descent_synth(
    initial_params, 
    target_spectrogram, 
    sample_rate=44100, 
    n_fft=2048, 
    hop_length=512, 
    max_iterations=100, 
    learning_rate=0.01, 
    tolerance=1e-5,
    duration=1.0
):
    """
    Manual gradient descent to optimize synthesizer parameters.
    
    :param initial_params: Initial synthesizer parameters [frequency, amplitude, attack, decay, sustain, release]
    :param target_spectrogram: Target spectrogram as the objective
    :param sample_rate: Sampling rate for audio
    :param n_fft: FFT size for spectrogram computation
    :param hop_length: Hop length for spectrogram computation
    :param max_iterations: Maximum number of iterations
    :param learning_rate: Step size for parameter adjustment
    :param tolerance: Stop if loss change is less than this value
    """
    params = np.array(initial_params, dtype=np.float32)
    deltas = np.array([learning_rate] * len(params))  # Small adjustments for each parameter
    prev_loss = float("inf")
    
    for iteration in range(max_iterations):
        current_waveform = synthesizer(parameters=params, sample_rate=sample_rate, duration=duration)
        _, _, current_spectrogram = convert_wave_to_spectrogram(current_waveform, sample_rate, n_fft, hop_length)
        
        # Calculate loss (MSE)
        current_loss = np.mean((current_spectrogram - target_spectrogram) ** 2)
        print(f"Iteration {iteration}: Loss = {current_loss}")
        
        # Stop if change in loss is negligible
        if abs(prev_loss - current_loss) < tolerance:
            print("Converged")
            break
        
        # Tweak each parameter
        for i in range(len(params)):
            best_loss = current_loss
            best_param = params[i]
            
            for direction in [-1, 1]:  # Test both decrease and increase
                params[i] = params[i] * (1+(direction * deltas[i])) # delta is a percentage
                new_waveform = synthesizer(params, sample_rate=sample_rate, duration=duration)
                _, _, new_spectrogram = convert_wave_to_spectrogram(new_waveform, sample_rate, n_fft, hop_length)
                new_loss = np.mean((new_spectrogram - target_spectrogram) ** 2)
                
                # Keep the parameter change if it improves the loss
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_param = params[i]
                
                # Revert the parameter for the next direction test
                params[i] -= direction * deltas[i]
            
            # Update the parameter to its best value
            params[i] = best_param

        print(F"params={params}")
        prev_loss = current_loss
    
    print("Final parameters:", params)
    return params

def main():
    # Target .wav file
    target_file = "slice_C5_104_84.wav"
    
    # Load the target .wav file and convert it to a spectrogram
    print("Loading target .wav file...")
    try:
        _, _, target_spectrogram, sample_rate, duration = load_wav_and_convert_to_spectrogram_scipy(target_file)
    except FileNotFoundError:
        print(f"Error: Target file '{target_file}' not found.")
        return

    # Initial synthesizer parameters: [frequency, amplitude, attack, decay, sustain, release]
    initial_params = [523, 1.0, 0.0001, 0.5, 0.01, 1.0]  # Reasonable defaults
    
    # Gradient descent parameters
    max_iterations = 10
    learning_rate = 0.05
    tolerance = 1e-5

    print("Starting gradient descent...")
    
    # Run the gradient descent optimizer
    optimized_params = gradient_descent_synth(
        initial_params,
        target_spectrogram,
        sample_rate=sample_rate,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        tolerance=tolerance,
        duration=duration
    )
    
    print(f"Optimized Parameters: {optimized_params}")
    
    # Generate the waveform using the optimized parameters
    print("Generating waveform with optimized parameters...")
    optimized_waveform = synthesizer(parameters=optimized_params, sample_rate=sample_rate, duration=duration)
    
    # Save the generated waveform to disk
    output_file = "optimized_output.wav"
    save_waveform_to_disk(optimized_waveform, sample_rate=sample_rate, output_path=output_file)

    print("Displaying the spectrogram of the optimized output...")
    # Display the spectrogram of the optimized output
    _, _, optimized_spectrogram = convert_wave_to_spectrogram(optimized_waveform, sample_rate)

    print("Optimization complete. Check the output wave file:", output_file)


if __name__ == "__main__":
    main()

