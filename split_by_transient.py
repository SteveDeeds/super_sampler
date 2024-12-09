import numpy as np
from scipy.io import wavfile
import os

def load_wave(input_file):
    """
    Loads a WAV file.
    
    Args:
        input_file (str): Path to the input WAV file.
        
    Returns:
        tuple: Sample rate (int) and audio data (numpy array).
    """
    sr, audio = wavfile.read(input_file)
    if len(audio.shape) > 1:  # Convert to mono if stereo
        audio = audio.mean(axis=1)
    audio = audio / np.max(np.abs(audio))  # Normalize audio
    return sr, audio

def find_split_locations(audio, sr, threshold=0.003, frame_size_ms=10, sample_length=2.0):
    """
    Finds split locations based on transient detection, rejecting short chunks.
    
    Args:
        audio (numpy array): Normalized audio data.
        sr (int): Sample rate of the audio.
        threshold (float): Amplitude change threshold to detect transients.
        frame_size_ms (int): Frame size in milliseconds for envelope calculation.
        sample_length (float): Minimum length of each sample in seconds.
        
    Returns:
        list: List of valid split points in samples.
    """
    frame_size = int(frame_size_ms * sr / 1000)  # Convert ms to samples
    min_sample_length = int(sample_length * sr)  # Minimum chunk length in samples

    # Calculate amplitude envelope (short-term energy)
    envelope = np.array([
        np.sqrt(np.mean(audio[i:i+frame_size]**2))
        for i in range(0, len(audio), frame_size)
    ])

    # Detect transient points
    transients = np.where(np.diff(envelope) > threshold)[0] * frame_size
    transients = transients.tolist()

    # Filter transients based on minimum sample length
    valid_splits = [0]  # Always include the start of the audio
    for i in range(len(transients)):
        if transients[i] - valid_splits[-1] >= min_sample_length:
            valid_splits.append(transients[i])

    # Add the end of the audio as the final split point
    if len(audio) - valid_splits[-1] >= min_sample_length:
        valid_splits.append(len(audio))

    return valid_splits

def split_and_save_audio(audio, sr, split_locations, output_folder):
    """
    Splits audio based on provided split locations and saves the chunks.

    Args:
        audio (numpy array): Normalized audio data.
        sr (int): Sample rate of the audio.
        split_locations (list): List of split points in samples, including start and end points.
        output_folder (str): Directory to save the output samples.
    """
    os.makedirs(output_folder, exist_ok=True)

    for i in range(len(split_locations) - 1):
        start_sample = split_locations[i]
        end_sample = split_locations[i + 1]

        # Extract the audio chunk
        chunk = audio[start_sample:end_sample]

        # Save the chunk as a separate WAV file
        output_file = os.path.join(output_folder, f"sample_{i+1}.wav")
        wavfile.write(output_file, sr, (chunk * 32767).astype(np.int16))

    print(f"Audio split into samples and saved in {output_folder}")

# Example usage
if __name__ == "__main__":
    input_wav = "ukulele.wav"  # Path to your WAV file
    output_dir = "output_samples"  # Directory to save the samples

    # Load the WAV file
    sample_rate, audio_data = load_wave(input_wav)

    # Find split locations with a minimum sample length of 2 seconds
    split_points = find_split_locations(audio_data, sample_rate, sample_length=2.0)

    # Split and save audio chunks
    split_and_save_audio(audio_data, sample_rate, split_points, output_dir)
