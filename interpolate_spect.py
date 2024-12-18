import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
import os
from scipy.io.wavfile import write, read as wavread
from scipy.signal import spectrogram, istft
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from wave_slice import analyze_frequency, change_pitch  # Import your function



def load_wav_files(directory, target_length_s=1.0):
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

    # First pass: Find the maximum length of the waveform
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            sample_rate, waveform = wavread(filepath)

            # Ensure waveform is mono
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            waveforms.append((filepath, waveform))

    normallized_pitch = (sample_rate/2048.0)*8

    # Second pass: Process each file with zero-padding to match the maximum length
    for filepath, waveform in waveforms:

        # store the actual frequency
        frequencies.append(analyze_frequency(waveform, sample_rate))
        # convert to a common frequency for analysis
        waveform = change_pitch(waveform, sample_rate, normallized_pitch)

        # Zero-pad the waveform if it is shorter than max_length
        if len(waveform) < target_length_s * sample_rate:
            waveform = np.pad(waveform, (0, int(target_length_s * sample_rate - len(waveform))), mode='constant')
        else:
            waveform = waveform[:int(target_length_s * sample_rate)]

        # Compute the spectrogram
        f, t, Sxx = spectrogram(waveform, fs=sample_rate, nperseg=2048, noverlap=1792)

        # Store spectrogram and associated frequencies/times
        spectrograms.append(Sxx)
        peak_values.append(np.max(waveform))

    # Convert to numpy arrays
    spectrograms = np.array(spectrograms, dtype=np.float32)
    frequencies = np.array(frequencies, dtype=np.float32)
    peak_values = np.array(peak_values, dtype=np.float32)

    return spectrograms, frequencies, peak_values, sample_rate, normallized_pitch

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

def clean_spectrogram(spectrogram, p=10):
    # Calculate the 10th percentile of the spectrogram
    percentile = np.percentile(spectrogram, p)
    
    # Subtract the percentile from the spectrogram
    spectrogram_processed = spectrogram - percentile
    
    # Set all negative values to zero
    spectrogram_processed[spectrogram_processed < 0] = 0
    
    return spectrogram_processed

def save_as_wav(output_path, spectrogram, sample_rate, pitch, normalized_pitch):
    """
    Saves spectrogram data as a .wav file by converting it back to time-domain audio.

    :param output_path: Path to save the .wav file.
    :param spectrogram: The magnitude spectrogram (time-frequency representation).
    :param sample_rate: The sample rate of the audio in Hz.
    """

    stretch_ratio = pitch / normalized_pitch
    zoom_factors = [1, stretch_ratio]
    spectrogram = zoom(spectrogram, zoom_factors, order=1)
    spectrogram = clean_spectrogram(spectrogram, 90)
    _, waveform = istft(spectrogram, fs=sample_rate, nperseg=2048, noverlap=1792)

    # Normalize the signal to the range [-1, 1] to prevent clipping
    waveform /= np.max(np.abs(waveform))
    waveform *= 0.999

    waveform = change_pitch(waveform, sample_rate, pitch)

    # Convert to 16-bit PCM format
    pcm_signal = np.int16(waveform * 32767)

    # Save as .wav
    write(output_path, sample_rate, pcm_signal)
    print(f"Saved .wav file to {output_path}")

def make_features(features):
    # Round the frequencies to the nearest 1 Hz
    rounded_frequencies = np.unique(np.round(features[:, 0]))
    
    # Extract peak amplitudes
    peak_amplitudes = features[:, 1]
    
    # Calculate the 10th and 90th percentiles
    amplitude_percentiles = [
        np.percentile(peak_amplitudes, 10),
        np.percentile(peak_amplitudes, 90),
    ]

    new_features = []
    for amp in amplitude_percentiles:
        for pitch in rounded_frequencies:
            new_features.append([pitch,amp])
    return np.array(new_features)


def fit_and_predict(learning_rate=0.2, d1=128, epochs=2000):
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
    spectrograms, frequencies, peak_values, sample_rate, normalized_pitch = load_wav_files(wav_directory)

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

    # Custom callback to save the best model to memory
    class InMemoryModelCheckpoint(Callback):
        def __init__(self, monitor='loss', mode='min', verbose=1):
            super(InMemoryModelCheckpoint, self).__init__()
            self.monitor = monitor
            self.mode = mode
            self.verbose = verbose
            self.best_model = None
            self.best_value = None
            
            # For 'min', initialize best_value with a large number, for 'max', a small number
            self.best_value = float('inf') if mode == 'min' else float('-inf')

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            current_value = logs.get(self.monitor)
            
            if current_value is not None:
                # Check if the current value is better
                if (self.mode == 'min' and current_value < self.best_value) or \
                (self.mode == 'max' and current_value > self.best_value):
                    self.best_value = current_value
                    self.best_model = tf.keras.models.clone_model(self.model)
                    self.best_model.set_weights(self.model.get_weights())
                    if self.verbose:
                        print(f"Epoch {epoch + 1}: {self.monitor} improved to {self.best_value}. Model saved to memory.")
                elif self.verbose:
                    print(f"Epoch {epoch + 1}: {self.monitor} did not improve.")

    # Usage
    checkpoint_callback = InMemoryModelCheckpoint(
        monitor='loss',
        mode='min',
        verbose=1
    )


    # # Define the callback to save the best model
    # checkpoint_callback = ModelCheckpoint(
    #     'best_model.keras',           # Path where the model will be saved
    #     monitor='loss',            # Metric to monitor
    #     mode='min',                # 'min' means the model with the lowest loss will be saved
    #     save_best_only=True,       # Save only the best model
    #     verbose=1                  # Print a message when the model is saved
    # )

    # Train the model
    history = model.fit(
        normalized_features,
        normalized_spectrograms,
        epochs=epochs,
        batch_size=8,
        verbose=1,
        callbacks=[checkpoint_callback]  # Include the callback
    )

    # Save the best model to disk after training
    if checkpoint_callback.best_model is not None:
        checkpoint_callback.best_model.save('best_model.keras')  # Save the model to disk
        print("Best model saved to 'best_model.keras'.")
    else:
        print("No best model was saved during training.")

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

    # # Optionally, after training, load the best model if needed
    # model = tf.keras.models.load_model('best_model.keras')

    # Interpolate a new spectrogram (real and imaginary parts)
    new_features = make_features(features)

    # Interpolate a new spectrogram for each feature
    normalized_new_features = (new_features - feature_mean) / feature_std
    normalized_interpolated_flat = model.predict(normalized_new_features)

    # Reshape predictions to a 3D array: (num_features, spectrogram_rows, spectrogram_cols)
    num_features = new_features.shape[0]
    spectrogram_shape = spectrograms.shape[1:]  # Shape of one spectrogram (e.g., rows, cols)
    interpolated_spectrograms = normalized_interpolated_flat.reshape((num_features, *spectrogram_shape))

    # Create "interpolated" folder if it doesn't exist
    output_folder = "interpolated"
    os.makedirs(output_folder, exist_ok=True)

    # Reverse normalization and save as wav files
    for i, feature in enumerate(new_features):
        interpolated_spectrogram = (interpolated_spectrograms[i] * spectrogram_std) + spectrogram_mean
        pitch = feature[0]  # Frequency in Hz
        file_name = os.path.join(output_folder, f"output_{learning_rate}_{d1}_{epochs}_peak_{int(feature[1])}_freq_{int(pitch)}.wav")
        spect = clean_spectrogram(interpolated_spectrogram)
        save_as_wav(file_name, spect, sample_rate, pitch, normalized_pitch)

    # #new_features = np.array([[523.25085, 6216.0205]])  # Example: frequency and normalized peak value
    # normalized_new_features = (new_features - feature_mean) / feature_std
    # normalized_interpolated_flat = model.predict(normalized_new_features)

    # # Reverse normalization on prediction
    # interpolated_flat = (normalized_interpolated_flat * spectrogram_std) + spectrogram_mean
    # interpolated_spectrogram = interpolated_flat.reshape(spectrograms.shape[1:])

    # # Save interpolated spectrogram as a .wav file
    # pitch = new_features[0,1]
    # save_as_wav(F"output_{learning_rate}_{d1}_{epochs}.wav", interpolated_spectrogram, sample_rate, pitch)

if __name__ == "__main__":
    fit_and_predict(1.0, 0, 1)
    fit_and_predict(0.001, 64, 50000)
