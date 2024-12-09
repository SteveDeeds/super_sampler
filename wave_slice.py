import os
import numpy as np
from scipy.io.wavfile import read, write
from scipy.fft import rfft, rfftfreq, fft, ifft
from scipy.signal import resample

class SliceInfo:
    """
    Represents information about a slice of a .wav file.
    """
    def __init__(self, index, frequency_hz, musical_note, peak_amplitude, velocity, slice_data):
        self.index = index
        self.frequency_hz = frequency_hz
        self.musical_note = musical_note
        self.peak_amplitude = peak_amplitude
        self.velocity = velocity
        self.slice_data = slice_data

    def __repr__(self):
        return (f"Slice {self.index}: {self.frequency_hz:.2f} Hz ({self.musical_note}), "
                f"Peak Amplitude: {self.peak_amplitude}, Velocity: {self.velocity}")

def frequency_to_note(freq):
    if freq <= 0:
        return "Unknown"
    
    A4 = 440.0
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_number = 12 * np.log2(freq / A4) + 69  # Corrected offset for A4 = MIDI 69
    note_index = int(round(note_number)) % 12
    octave = (int(round(note_number)) // 12) - 1
    return f"{notes[note_index]}{octave}"

def analyze_frequency(wave_data, sample_rate):
    """
    Analyzes the dominant frequency in a wave slice using FFT.
    """
    N = len(wave_data)
    yf = rfft(wave_data)
    xf = rfftfreq(N, 1 / sample_rate)
    idx = np.argmax(np.abs(yf))
    return xf[idx]

def calculate_velocity(peak_amplitude, max_amplitude):
    """
    Calculates velocity relative to the loudest slice, scaled to a max of 127.
    """
    if max_amplitude == 0:
        return 0
    return int((peak_amplitude / max_amplitude) * 127)

def change_pitch(audio_data, sample_rate, target_frequency):
    """
    Changes the pitch of the audio data by adjusting the sample rate.
    
    Parameters:
    - audio_data: np.ndarray, the waveform data as a numpy array.
    - sample_rate: int, the sample rate of the audio.
    - target_frequency: float, the desired frequency in Hz.
    
    Returns:
    - np.ndarray: The pitch-adjusted audio data.
    """
    # Analyze the current dominant frequency
    current_frequency = analyze_frequency(audio_data, sample_rate)

    if current_frequency <= 0:
        raise ValueError("The audio signal contains no detectable frequency.")
    
    # Calculate the pitch adjustment factor
    pitch_factor = current_frequency / target_frequency
    
    # Change the pitch by adjusting the sample rate
    resampled_audio = resample(audio_data, int(len(audio_data) * pitch_factor))
    
    # Return the pitch-adjusted audio
    return resampled_audio.astype(np.float32)

def tune_to_nearest_note(audio_data, sample_rate):
    """
    Tunes an audio sample to the nearest musical note by adjusting its pitch.

    Parameters:
    - audio_data: np.ndarray, the waveform data as a numpy array.
    - sample_rate: int, the sample rate of the audio.

    Returns:
    - tuned_audio_data: np.ndarray, the pitch-corrected audio data.
    - target_note: str, the name of the target musical note.
    """

    # Step 1: Analyze the dominant frequency in the audio
    dominant_freq = analyze_frequency(audio_data, sample_rate)
    
    if dominant_freq <= 0:
        return audio_data, "Unknown"
    
    # Step 2: Find the nearest musical note and its frequency
    A4 = 440.0  # Reference frequency for A4
    note_number = 12 * np.log2(dominant_freq / A4) + 49
    target_freq = A4 * 2 ** ((round(note_number) - 49) / 12)
    
    # Step 3: Adjust pitch to match the nearest note
    tuned_audio_data = change_pitch(audio_data, sample_rate, target_freq)
    
    return tuned_audio_data

import numpy as np

def generate_wavetable(audio_data, wavelet_length=2048, number_of_waves=256):
    """
    Generates a wavetable from audio data with a uniform random phase applied to all wavelets.

    Parameters:
    - audio_data: np.ndarray, the waveform data as a numpy array.
    - wavelet_length: int, the length of each wavelet in the wavetable.
    - number_of_waves: int, the number of wavelets to generate.

    Returns:
    - np.ndarray: The concatenated wavetable as a single array.
    """
    total_samples = len(audio_data)

    # Generate a window for the specified wavelet length
    # window = np.blackman(wavelet_length)
    window = np.ones(wavelet_length)

    # phase = []
    # for _ in range(wavelet_length):
    #     angle = np.random.uniform(0, 2 * np.pi)  # Random angle between 0 and 2*pi radians
    #     z = np.cos(angle) + 1j * np.sin(angle)  # Complex number representation
    #     phase.append(z)        

    # zero phase
    phase = np.ones(wavelet_length)* 1j
    phase[len(phase)//2:] = -1j

    # Initialize the wavetable and RMS values list
    wavetable = []
    rms_values = []

    last_wave_start = total_samples - wavelet_length
    wave_step = last_wave_start // number_of_waves
    start_sample = 0

    for i in range(number_of_waves):
        end_sample = start_sample + wavelet_length

        # Extract a slice (wave) of the audio data
        wave = audio_data[start_sample:end_sample]

        # Apply the window
        wave_windowed = wave * window

        # Apply the phase and convert back to time domain
        wave_fft = fft(wave_windowed)
        mag_fft = np.abs(wave_fft)
        wave_fft_phased = mag_fft * phase
        ifft_wave = ifft(wave_fft_phased)
        wave_with_phase = np.real(ifft_wave)

        # Calculate the RMS power of the wave
        rms_power = np.sqrt(np.mean(wave_with_phase**2))
        rms_values.append(rms_power)

        # Append the phase-modulated wave to the wavetable
        wavetable.append(wave_with_phase)

        # Make sure we don't go over range
        start_sample = min(start_sample + wave_step, last_wave_start)

    # # Normalize each wavelet by its RMS value
    # for i in range(len(wavetable)):
    #     wavetable[i] = wavetable[i] * (0.05 / rms_values[i])

    # Concatenate all the wavelets into a single 1D array
    full_wavetable = np.concatenate(wavetable)

    return full_wavetable


def load_wav(file_path):
    """
    Loads a .wav file, preprocesses it by normalizing and converting to mono if necessary.

    Parameters:
    - file_path: str, the path to the .wav file.

    Returns:
    - sample_rate: int, the sample rate of the .wav file.
    - wave_data: np.ndarray, the preprocessed wave data.
    """
    sample_rate, wave_data = read(file_path)

    # Normalize data if needed
    if wave_data.dtype == np.int16:
        wave_data = (wave_data / 32768.0).astype(np.float32)
    elif wave_data.dtype == np.int32:
        wave_data = (wave_data / 2147483648.0).astype(np.float32)
    elif wave_data.dtype == np.uint8:
        wave_data = ((wave_data - 128) / 128.0).astype(np.float32)

    # Convert stereo to mono if necessary
    if len(wave_data.shape) > 1:
        wave_data = wave_data.mean(axis=1)

    return sample_rate, wave_data

def split_wav(wave_data, sample_rate, num_slices, output_dir = "output", samples_dir="samples", wavetables_dir="wavetables"):
    """
    Splits a .wav file into slices and generates wavetables for each slice.

    Parameters:
    - wave_data: np.ndarray, the preprocessed wave data.
    - sample_rate: int, the sample rate of the audio.
    - num_slices: int, the number of slices to split the file into.
    - samples_dir: str, directory to save the slices.
    - wavetables_dir: str, directory to save the wavetables.

    Returns:
    - List[SliceInfo]: A list of SliceInfo objects for each slice.
    """
    os.makedirs(os.path.join(output_dir,samples_dir), exist_ok=True)
    os.makedirs(os.path.join(output_dir,wavetables_dir), exist_ok=True)
    slices_info = []

    num_frames = len(wave_data)
    frames_per_slice = num_frames / num_slices
    peak_amplitudes = []

    # Step 1: Process all slices to collect peak amplitudes
    for i in range(num_slices):
        start = int(i * frames_per_slice)
        end = int(min(start + frames_per_slice, num_frames))
        slice_data = wave_data[start:end]
        peak_amplitudes.append(np.max(np.abs(slice_data)))

    # Step 2: Determine the maximum peak amplitude
    max_amplitude = max(peak_amplitudes)

    # Step 3: Process slices with velocity calculations
    for i in range(num_slices):
        start = int(i * frames_per_slice)
        end = int(min(start + frames_per_slice, num_frames))
        slice_data = wave_data[start:end]

        # Tune the note to the nearest pitch
        slice_data = tune_to_nearest_note(slice_data, sample_rate)

        # Analyze frequency, determine musical note, and calculate velocity
        dominant_freq = analyze_frequency(slice_data, sample_rate)
        note = frequency_to_note(dominant_freq)
        peak_amplitude = peak_amplitudes[i]
        velocity = calculate_velocity(peak_amplitude, max_amplitude)

        # Save the slice in the "samples" directory
        upper_velocity = min(127, velocity + 10)
        lower_velocity = max(0, velocity - 10)
        slice_filename = os.path.join(output_dir, samples_dir, f"slice_{note}_{upper_velocity}_{lower_velocity}.wav")
        write(slice_filename, sample_rate, (slice_data * 32767).astype(np.int16))

        # Generate and save the wavetable in the "wavetables" directory
        slice_data = change_pitch(slice_data, sample_rate, (sample_rate/2048)*8)
        wavetable = generate_wavetable(slice_data)
        wavetable_filename = os.path.join(output_dir, wavetables_dir, f"wavetable_{note}_{upper_velocity}_{lower_velocity}.wav")
        write(wavetable_filename, sample_rate, (wavetable * 32767).astype(np.int16))

        # Store slice information
        slice_info = SliceInfo(
            index=i + 1,
            frequency_hz=dominant_freq,
            musical_note=note,
            peak_amplitude=peak_amplitude,
            velocity=velocity,
            slice_data=slice_data
        )
        slices_info.append(slice_info)
        print(slice_info)

    return slices_info

# Main Function
def main():
    file_path = "ukulele.wav"
    num_slices = 40

    # Load the .wav file
    sample_rate, wave_data = load_wav(file_path)

    # Split the wave data into slices
    slices = split_wav(wave_data, sample_rate, num_slices)

    return slices


# Example usage
if __name__ == "__main__":
    main()