import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from scipy.io.wavfile import write, read as wavread
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt


def load_wav_files(directory):
    frequencies = []
    amplitudes = []
    fft_data = []

    max_freq_bins = 0
    samplerate = None

    # First pass: Determine the maximum number of frequency bins
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            samplerate, waveform = wavread(filepath)

            # Ensure waveform is mono
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            # Normalize waveform
            waveform = waveform / np.max(np.abs(waveform))

            # Compute FFT
            fft_result = fft(waveform)
            max_freq_bins = max(max_freq_bins, len(fft_result))

    # Second pass: Process each file with zero-padding in the frequency dimension
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            print(F"loading {filepath}")
            samplerate, waveform = wavread(filepath)

            # Ensure waveform is mono
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            # Compute FFT
            fft_result = fft(waveform)

            # Separate into real and imaginary parts
            real_part = np.real(fft_result)
            imag_part = np.imag(fft_result)

            # Zero-pad FFT results
            padded_real = np.pad(real_part, (0, max_freq_bins - len(real_part)), mode='constant')
            padded_imag = np.pad(imag_part, (0, max_freq_bins - len(imag_part)), mode='constant')

            # Combine real and imaginary parts
            combined_fft = np.stack([padded_real, padded_imag], axis=-1)

            # Measure peak amplitude and frequency
            peak_amplitude = np.max(np.abs(waveform))
            peak_frequency_idx = np.argmax(np.abs(fft_result))
            peak_frequency = peak_frequency_idx * samplerate / len(waveform)

            # Store results
            print(F"freq={peak_frequency} amplitude={peak_amplitude}")
            frequencies.append(peak_frequency)
            amplitudes.append(peak_amplitude)
            fft_data.append(combined_fft)

    # Convert to numpy arrays
    frequencies = np.array(frequencies, dtype=np.float32)
    amplitudes = np.array(amplitudes, dtype=np.float32)
    fft_data = np.array(fft_data)

    return frequencies, amplitudes, fft_data, samplerate

def create_model(spectrogram_shape, learning_rate=0.2, d1=128):
    """
    Creates a TensorFlow model to predict spectrogram data from frequency and amplitude.

    :param spectrogram_shape: Shape of the spectrogram output (real and imaginary parts).
    :return: A compiled TensorFlow model.
    """
    output_size = np.prod(spectrogram_shape)

    input_layer = layers.Input(shape=(2,))  # Input: frequency_hz and peak_amplitude
    if d1 > 0:
        x = layers.Dense(d1, activation='relu')(input_layer)
    else:
        x = input_layer
    output_layer = layers.Dense(output_size, activation='linear')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # Default is 0.001; increase as needed
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def save_as_wav(output_path, interpolated_fft, samplerate):
    """
    Saves interpolated FFT data (complex values) as a .wav file.

    :param output_path: Path to save the .wav file.
    :param interpolated_fft: 1D array of complex FFT data.
    :param samplerate: The sample rate of the audio in Hz.
    """
    # Reconstruct time-domain signal using inverse FFT
    time_domain_signal = np.real(ifft(interpolated_fft))

    # Normalize the signal to the range [-1, 1] to prevent clipping
    time_domain_signal /= np.max(np.abs(time_domain_signal))

    # Convert to 16-bit PCM format
    pcm_signal = np.int16(time_domain_signal * 32767)

    # Save as .wav
    write(output_path, samplerate, pcm_signal)
    print(f"Saved .wav file to {output_path}")


def fit_and_predict(learning_rate=0.2, d1=128,epochs=5000):
    # Directory containing .wav files
    print("starting...")
    wav_directory = "./train"

    # Load and process .wav files into spectrograms (real and imaginary parts)
    frequencies, amplitudes, spectrogram_data, sample_rate = load_wav_files(wav_directory)

    # Flatten spectrograms for training
    flattened_spectrograms = spectrogram_data.reshape(spectrogram_data.shape[0], -1)

    # Normalize the spectrograms along features (axis=1)
    spectrogram_mean = flattened_spectrograms.mean()
    spectrogram_std = flattened_spectrograms.std()
    normalized_spectrograms = (flattened_spectrograms - spectrogram_mean) / spectrogram_std

    # Normalize the features
    features = np.stack([frequencies, amplitudes], axis=1)
    feature_mean = features.mean(axis=0, keepdims=True)
    feature_std = features.std(axis=0, keepdims=True)
    normalized_features = (features - feature_mean) / feature_std

    # Train the model
    model = create_model(normalized_spectrograms.shape[1:],learning_rate=learning_rate, d1=d1)

    # Define the callback to save the best model (with the lowest loss)
    checkpoint_callback = ModelCheckpoint(
        'best_model.keras',           # Path where the model will be saved
        monitor='loss',            # Metric to monitor
        mode='min',                # 'min' means the model with the lowest loss will be saved
        save_best_only=True,       # Save only the best model (lowest loss)
        verbose=1                  # Print a message when the model is saved
    )

    # Fit the model with the checkpoint callback
    history = model.fit(
        normalized_features,
        normalized_spectrograms,
        epochs=epochs,
        batch_size=8,
        verbose=1,
        callbacks=[checkpoint_callback]  # Include the callback here
    )

    # Access the loss values from the history object
    loss = history.history['loss']

    # Plot the loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss) + 1), loss, label='Loss', color='blue')
    plt.yscale('log')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(F"loss_{learning_rate}_{d1}_{epochs}.png")

    # Optionally, after training, load the best model if needed
    best_model = tf.keras.models.load_model('best_model.keras')

    # Interpolate a new spectrogram (real and imaginary parts)
    new_features = np.array([[329.63034, 17155.0]])
    normalized_new_features = (new_features - feature_mean) / feature_std
    normalized_interpolated_flat = best_model.predict(normalized_new_features)

    # Reverse normalization on prediction
    interpolated_flat = (normalized_interpolated_flat * spectrogram_std) + spectrogram_mean
    interpolated_fft = interpolated_flat.reshape(spectrogram_data.shape[1:])

    # Separate real and imaginary components
    real_part = interpolated_fft[..., 0]
    imag_part = interpolated_fft[..., 1]

    # Combine into complex FFT data
    complex_fft = real_part + 1j * imag_part

    # Save interpolated spectrogram as a .wav file
    save_as_wav(F"output_{learning_rate}_{d1}_{epochs}.wav", complex_fft, sample_rate)

if __name__ == "__main__":
    fit_and_predict(1.0, 0, 1)
    fit_and_predict(0.01, 16)    
    fit_and_predict(0.01, 32)
    fit_and_predict(0.01, 64)
    fit_and_predict(0.01, 128)
