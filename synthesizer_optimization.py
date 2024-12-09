import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read as wavfile_read
from scipy.signal import spectrogram
from scipy.signal import istft
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# Scipy-based functions
def convert_wave_to_spectrogram(waveform, sample_rate=44100, n_fft=2048, hop_length=512):
    frequencies, times, S = spectrogram(waveform, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length, mode='magnitude')
    S_db = 10 * np.log10(S + 1e-10)  # Convert to decibels and avoid log(0)
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


# Function to plot spectrogram
def plot_spectrogram_scipy(frequencies, times, S_db):
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, S_db, shading='gouraud', cmap='inferno')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

# Envelope function (ADSR)
def envelope(t, attack, decay, sustain, release, duration):
    envelope = np.zeros_like(t)
    attack_end = attack
    decay_end = attack + decay
    release_start = duration - release

    # Attack phase
    envelope[(t >= 0) & (t < attack_end)] = t[(t >= 0) & (t < attack_end)] / attack
    # Decay phase
    envelope[(t >= attack_end) & (t < decay_end)] = 1 - ((1 - sustain) * (t[(t >= attack_end) & (t < decay_end)] - attack_end) / decay)
    # Sustain phase
    envelope[(t >= decay_end) & (t < release_start)] = sustain
    # Release phase
    envelope[(t >= release_start) & (t <= duration)] = sustain * (1 - (t[(t >= release_start) & (t <= duration)] - release_start) / release)

    return envelope

def synthesizer(parameters, sample_rate=44100, target_spectrogram=None, duration=None, n_fft=2048, hop_length=512):
    if target_spectrogram is not None and duration is None:
        duration = target_spectrogram.shape[1] * (n_fft - hop_length) / sample_rate
    frequency, amplitude, attack, decay, sustain, release = parameters
    t = tf.linspace(0.0, duration, int(sample_rate * duration))
    env = envelope(t, attack, decay, sustain, release, duration)
    waveform = amplitude * env * tf.sin(2 * np.pi * frequency * t)  # TensorFlow operation
    return waveform


# TensorFlow-compatible synthesizer wrapper
def tf_synthesizer(parameters, sample_rate=44100, duration=1.0):
    parameters_np = parameters.numpy()
    waveform = synthesizer(parameters_np, sample_rate=sample_rate, duration=duration)
    return tf.convert_to_tensor(waveform, dtype=tf.float32)

def numerical_gradient(f, parameters, epsilon=1e-5):
    """
    Computes the gradient of a scalar-valued function `f` using the finite difference method.
    
    Args:
        f: A function that takes parameters and returns a scalar loss.
        parameters: A list or array of parameters (can be TensorFlow tensors or NumPy arrays).
        epsilon: A small step size for finite difference.
        
    Returns:
        grad: A numpy array of the same shape as `parameters` representing the gradient.
    """
    # Ensure parameters are a NumPy array
    grad = tf.zeros_like(parameters)
    
    for i in range(len(parameters)):
        params_plus = parameters.copy()
        params_minus = parameters.copy()
        
        params_plus[i] += epsilon
        params_minus[i] -= epsilon
        
        # Evaluate the function at slightly perturbed parameter values
        loss_plus = f(params_plus)
        loss_minus = f(params_minus)
        
        # Compute the gradient for this dimension
        grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
    
    return grad


def loss_function(parameters,duration):
    waveform = synthesizer(parameters, duration=duration)
    _, _, generated_spectrogram = convert_wave_to_spectrogram(waveform)
    return np.mean((generated_spectrogram - target_spectrogram.numpy()) ** 2)


# Optimization routine
def optimize_parameters(target_spectrogram, initial_parameters, duration, learning_rate=0.01, steps=1000):
    # Convert initial parameters to tf.Variable so TensorFlow can optimize them
    parameters = tf.Variable(initial_parameters, dtype=tf.float32)
    
    # Use Adam optimizer (you can adjust learning rate as needed)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    for step in range(steps):
        # with tf.GradientTape() as tape:
        #     # Watch the parameters since they're tf.Variable now
        #     tape.watch(parameters)

        #     # Calculate the loss using the current parameters
        #     loss = loss_function(parameters, target_spectrogram, duration=duration)
        #     print(F"loss = {loss}")

        # Compute gradients of the loss with respect to the parameters
        gradients = numerical_gradient(loss_function, (parameters,duration))
        print(F"gradients = {gradients}")
        print(F"[parameters] = {[parameters]}")


        # Apply the gradients to the parameters (update them using gradient descent)
        optimizer.apply_gradients(zip(gradients, [parameters]))

        # # Log progress every 100 steps
        # if step % 100 == 0:
        #     print(f"Step {step}, Loss: {loss.numpy()}, Parameters: {parameters.numpy()}")

    return parameters.numpy()



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


# Main workflow
if __name__ == "__main__":
    # Load target spectrogram from a .wav file
    file_path = "slice_C5_104_84.wav"  # Replace with your target .wav file path
    frequencies, times, S_db, sr, duration = load_wav_and_convert_to_spectrogram_scipy(file_path)
    target_spectrogram = tf.convert_to_tensor(S_db / np.max(S_db), dtype=tf.float32)
    
    # Initial parameters (frequency, amplitude, attack, decay, sustain, release)
    initial_parameters = np.array([440.0, 1.0, 0.1, 0.2, 0.7, 0.5], dtype=np.float32)
    
    # Optimize parameters
    optimized_parameters = optimize_parameters(target_spectrogram, initial_parameters, duration, learning_rate=0.01, steps=1000)
    
    print("Optimized Parameters:", optimized_parameters)
    
    # Regenerate spectrogram from the optimized parameters
    generated_spectrogram = synthesizer(optimized_parameters, sample_rate=sr, duration=duration)

    plt.figure(figsize=(12, 6))

    # Plot target spectrogram
    plt.subplot(1, 2, 1)
    plot_spectrogram_scipy(frequencies, times, target_spectrogram.numpy())
    plt.title("Target Spectrogram")

    # Plot generated spectrogram
    plt.subplot(1, 2, 2)
    generated_frequencies, generated_times, generated_spectrogram_db = convert_wave_to_spectrogram(generated_spectrogram, sample_rate=sr)
    plot_spectrogram_scipy(generated_frequencies, generated_times, generated_spectrogram_db)
    plt.title("Generated Spectrogram")

    plt.show()

    # Convert the generated spectrogram back to a waveform
    generated_waveform = convert_spectrogram_to_waveform(generated_spectrogram_db, sample_rate=sr, n_fft=2048, hop_length=512)
    
    # Save the generated waveform to disk
    save_waveform_to_disk(generated_waveform, sample_rate=sr, output_path="generated_output.wav")

