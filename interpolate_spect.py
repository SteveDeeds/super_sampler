import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from scipy.io.wavfile import write, read as wavread
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from wave_slice import analyze_frequency, change_pitch  # Import your function



def load_wav_files(directory, target_length=None, target_time_bins=100, target_freq_bins=513):
    """
    Load .wav files from a directory and process them into spectrograms, 
    zero-padded to a fixed size and length of the longest file.

    :param directory: Path to the directory containing .wav files.
    :param target_length: The length of the waveform to pad to (None for no padding, otherwise use the max length).
    :param target_time_bins: Target number of time bins for the spectrogram.
    :param target_freq_bins: Target number of frequency bins for the spectrogram.
    :return: Tuple of spectrograms, frequencies, and times.
    """
    spectrograms = []
    frequencies = []
    peak_values = []
    waveforms = []
    sample_rate = None
    max_length = 0

    # First pass: Find the maximum length of the waveform
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            sample_rate, waveform = wavread(filepath)

            # Ensure waveform is mono
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            # Track the longest waveform length
            max_length = max(max_length, len(waveform))

            waveforms.append((filepath, waveform))

    # Second pass: Process each file with zero-padding to match the maximum length
    for filepath, waveform in waveforms:
        # Zero-pad the waveform if it is shorter than max_length
        if len(waveform) < max_length:
            waveform = np.pad(waveform, (0, max_length - len(waveform)), mode='constant')

        # Compute the spectrogram
        f, t, Sxx = spectrogram(waveform, fs=sample_rate, nperseg=1024, noverlap=512)

        # Pad or truncate the spectrogram to match target_freq_bins and target_time_bins
        if Sxx.shape[0] > target_freq_bins:
            Sxx = Sxx[:target_freq_bins, :]  # Truncate frequency bins
        else:
            Sxx = np.pad(Sxx, ((0, target_freq_bins - Sxx.shape[0]), (0, 0)), mode='constant')  # Pad frequency bins

        if Sxx.shape[1] > target_time_bins:
            Sxx = Sxx[:, :target_time_bins]  # Truncate time bins
        else:
            Sxx = np.pad(Sxx, ((0, 0), (0, target_time_bins - Sxx.shape[1])), mode='constant')  # Pad time bins

        # Store spectrogram and associated frequencies/times
        spectrograms.append(Sxx)
        frequencies.append(analyze_frequency(waveform, sample_rate))
        peak_values.append(np.max(waveform))

    # Convert to numpy arrays
    spectrograms = np.array(spectrograms, dtype=np.float32)
    frequencies = np.array(frequencies, dtype=np.float32)
    peak_values = np.array(peak_values, dtype=np.float32)

    return spectrograms, frequencies, peak_values, sample_rate

def create_model(spectrogram_shape, learning_rate=0.2, d1=128):
    """
    Creates a TensorFlow model to predict spectrogram data.

    :param spectrogram_shape: Shape of the spectrogram output (time-frequency bins).
    :return: A compiled TensorFlow model.
    """
    output_size = np.prod(spectrogram_shape)

    input_layer = layers.Input(shape=(2,))  # Input: frequency and amplitude (customize as needed)
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

def save_as_wav(output_path, spectrogram, sample_rate):
    """
    Saves spectrogram data as a .wav file by converting it back to time-domain audio.

    :param output_path: Path to save the .wav file.
    :param spectrogram: The magnitude spectrogram (time-frequency representation).
    :param sample_rate: The sample rate of the audio in Hz.
    """
    # Reconstruct time-domain signal from spectrogram using inverse STFT
    from scipy.signal import istft

    _, waveform = istft(spectrogram, fs=sample_rate, nperseg=1024, noverlap=512)

    # Normalize the signal to the range [-1, 1] to prevent clipping
    waveform /= np.max(np.abs(waveform))

    # Convert to 16-bit PCM format
    pcm_signal = np.int16(waveform * 32767)

    # Save as .wav
    write(output_path, sample_rate, pcm_signal)
    print(f"Saved .wav file to {output_path}")

def fit_and_predict(learning_rate=0.2, d1=128, epochs=5000):
    """
    Fits the model to the spectrogram data and predicts a new spectrogram using input features.
    :param learning_rate: Learning rate for the optimizer.
    :param d1: Size of the dense layer (set to 0 to disable it).
    :param epochs: Number of training epochs.
    """
    # Directory containing .wav files
    print("Starting model training...")
    wav_directory = "./train"

    # Load and process .wav files into spectrograms and additional features
    spectrograms, frequencies, peak_values, sample_rate = load_wav_files(wav_directory)

    # Flatten spectrograms for training (time-frequency bins)
    flattened_spectrograms = spectrograms.reshape(spectrograms.shape[0], -1)

    # Normalize the spectrograms
    spectrogram_mean = flattened_spectrograms.mean()
    spectrogram_std = flattened_spectrograms.std()
    normalized_spectrograms = (flattened_spectrograms - spectrogram_mean) / spectrogram_std

    # Combine features (frequencies and peak_values) into a single feature set
    features = np.stack([frequencies, peak_values], axis=1)

    # Normalize the features
    feature_mean = features.mean(axis=0, keepdims=True)
    feature_std = features.std(axis=0, keepdims=True)
    normalized_features = (features - feature_mean) / feature_std

    # Create and compile the model
    model = create_model(normalized_spectrograms.shape[1:], learning_rate=learning_rate, d1=d1)

    # Define the callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        'best_model.keras',           # Path where the model will be saved
        monitor='loss',            # Metric to monitor
        mode='min',                # 'min' means the model with the lowest loss will be saved
        save_best_only=True,       # Save only the best model
        verbose=1                  # Print a message when the model is saved
    )

    # Train the model
    history = model.fit(
        normalized_features,
        normalized_spectrograms,
        epochs=epochs,
        batch_size=8,
        verbose=1,
        callbacks=[checkpoint_callback]  # Include the callback
    )

    # Plot the loss over epochs
    loss = history.history['loss']
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss) + 1), loss, label='Loss', color='blue')
    plt.yscale('log')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(F"loss_{learning_rate}_{d1}_{epochs}.png")

    # Optionally, after training, load the best model if needed
    best_model = tf.keras.models.load_model('best_model.keras')

    # Interpolate a new spectrogram (real and imaginary parts)
    new_features = np.array([[329.63034, 0.5]])  # Example: frequency and normalized peak value
    normalized_new_features = (new_features - feature_mean) / feature_std
    normalized_interpolated_flat = best_model.predict(normalized_new_features)

    # Reverse normalization on prediction
    interpolated_flat = (normalized_interpolated_flat * spectrogram_std) + spectrogram_mean
    interpolated_spectrogram = interpolated_flat.reshape(spectrograms.shape[1:])

    # Save interpolated spectrogram as a .wav file
    save_as_wav(F"output_{learning_rate}_{d1}_{epochs}.wav", interpolated_spectrogram, sample_rate)

if __name__ == "__main__":
    fit_and_predict(1.0, 0, 1)
    fit_and_predict(0.001, 0)
    # fit_and_predict(0.01, 0)
    # fit_and_predict(0.1, 0)
    # fit_and_predict(1.0, 0)
