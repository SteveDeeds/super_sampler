import numpy as np
import matplotlib.pyplot as plt
from wave_slice import process_slices
from scipy.ndimage import maximum_filter1d
from SliceInfo import load_wav

def find_split_locations(audio, sr, sensitivity=0.5, frame_size_ms=10, sample_length=2.0, plot=False):
    """
    Finds split locations based on transient detection, rejecting short chunks, and optionally plots the wave data, splits, and peak envelope.
    
    Args:
        audio (numpy array): Normalized audio data.
        sr (int): Sample rate of the audio.
        threshold (float): Amplitude change threshold to detect transients.
        frame_size_ms (int): Frame size in milliseconds for envelope calculation.
        sample_length (float): Minimum length of each sample in seconds.
        plot (bool): If True, plots the waveform, peak envelope, and valid splits. Default is False.
        
    Returns:
        list: List of valid split points in samples.
    """
    frame_size = int(frame_size_ms * sr / 1000)  # Convert ms to samples
    min_sample_length = int(sample_length * sr)  # Minimum chunk length in samples

    padded_audio = np.pad(audio, (frame_size, 0), mode='constant', constant_values=0)

    # Calculate the peak envelope using a 1D maximum filter
    envelope = maximum_filter1d(np.abs(padded_audio), size=frame_size, mode='constant')

    envelope = envelope[frame_size//2:-frame_size//2]

    # Detect transient points
    transients = []
    max_envelope = np.max(envelope)
    for i in range(frame_size, len(envelope)):
        if (envelope[i] >= envelope[i - frame_size] / sensitivity) and (envelope[i]>max_envelope*0.005):
            transients.append(i)

    transients = np.array(transients)

    # Filter transients based on minimum sample length
    valid_splits = [0]  # Always include the start of the audio
    for i in range(len(transients)):
        if transients[i] - valid_splits[-1] >= min_sample_length:
            valid_splits.append(transients[i])

    # Add the end of the audio as the final split point
    if len(audio) - valid_splits[-1] >= min_sample_length:
        valid_splits.append(len(audio))

    # Plot if requested
    if plot:
        time = np.linspace(0, len(audio) / sr, len(audio))
        frame_times = np.linspace(0, len(audio) / sr, len(envelope))

        plt.figure(figsize=(14, 6))

        # Plot the waveform
        plt.plot(time, audio, label="Waveform", alpha=0.6)

        # Plot the peak envelope
        plt.plot(frame_times, envelope, label="Peak Envelope", color="orange", linewidth=2)

        # Plot valid splits
        for split in valid_splits:
            plt.axvline(x=split / sr, color="red", linestyle="--", label="Split" if split == valid_splits[0] else None)

        plt.title("Waveform, Peak Envelope, and Split Points")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()
        plt.show()
        plt.waitforbuttonpress()

        slice_ranges = [(valid_splits[i], valid_splits[i + 1]) for i in range(len(valid_splits) - 1)]

    return slice_ranges

# def split_and_save_audio(audio, sr, split_locations, output_folder):
#     """
#     Splits audio based on provided split locations and saves the chunks.

#     Args:
#         audio (numpy array): Normalized audio data.
#         sr (int): Sample rate of the audio.
#         split_locations (list): List of split points in samples, including start and end points.
#         output_folder (str): Directory to save the output samples.
#     """
#     os.makedirs(output_folder, exist_ok=True)

#     for i in range(len(split_locations) - 1):
#         start_sample = split_locations[i]
#         end_sample = split_locations[i + 1]

#         # Extract the audio chunk
#         chunk = audio[start_sample:end_sample]

#         # Save the chunk as a separate WAV file
#         output_file = os.path.join(output_folder, f"sample_{i+1}.wav")
#         wavfile.write(output_file, sr, (chunk * 32767).astype(np.int16))

#     print(f"Audio split into samples and saved in {output_folder}")

# Example usage
if __name__ == "__main__":
    input_wav = "CP33-EPiano1.wav"  # Path to your WAV file
    output_dir = "output_samples"  # Directory to save the samples

    # Load the WAV file
    sample_rate, audio_data = load_wav(input_wav)

    # Find split locations with a minimum sample length of 2 seconds
    split_points = find_split_locations(audio_data, sample_rate, sample_length=2.0, plot=True, frame_size_ms=80)

    # Split and save audio chunks
    process_slices(wave_data=audio_data, sample_rate=sample_rate, slice_ranges=split_points, output_dir=output_dir)
