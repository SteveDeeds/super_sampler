import os
import numpy as np
from scipy.io.wavfile import read, write
from scipy.fft import rfft, rfftfreq, fft, ifft
from scipy.signal import resample
from split_by_transient import find_split_locations

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
    note_number = 12 * np.log2(freq / A4) + 69
    note_index = int(round(note_number)) % 12
    octave = (int(round(note_number)) // 12) - 1
    return f"{notes[note_index]}{octave}"

# def analyze_frequency(wave_data, sample_rate):
#     """
#     Analyzes the dominant frequency in a wave slice using FFT.
#     """
#     N = len(wave_data)
#     yf = rfft(wave_data)
#     xf = rfftfreq(N, 1 / sample_rate)
#     idx = np.argmax(np.abs(yf))
#     return xf[idx]

def analyze_frequency(signal, sample_rate):
    N = len(signal)
    spectrum = np.abs(np.fft.fft(signal, n=N))[:N // 2]

    # Downsample harmonics and ensure equal lengths
    harmonics = [spectrum[::i] for i in range(1, 10)]
    min_length = min(len(h) for h in harmonics)
    harmonics = [h[:min_length] for h in harmonics]  # Trim all harmonics to the shortest length

    # Harmonic Product Spectrum calculation
    hps = np.sum(harmonics, axis=0)
    peak_index = np.argmax(hps)

    # Convert the peak index to frequency
    freq = sample_rate * peak_index / N
    return freq


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

# def generate_wavetable(audio_data, sample_rate, wavelet_length=2048, number_of_waves=256):
#     """
#     Generate a wavetable from the input audio data.
    
#     Parameters:
#         audio_data (numpy array): The input audio signal.
#         sample_rate (int): The sample rate of the audio signal.
#         wavelet_length (int): The length of each wavelet in the wavetable.
#         number_of_waves (int): The number of waveforms in the wavetable.
    
#     Returns:
#         numpy array: The generated wavetable of shape (number_of_waves, wavelet_length).
#     """
#     # Step 1: Determine the pitch using the analyze_frequency function
#     fundamental_freq = analyze_frequency(audio_data, sample_rate)
#     if fundamental_freq <= 0:
#         raise ValueError("Could not determine a valid fundamental frequency.")
    
#     # Step 2: Calculate the period of the fundamental frequency
#     period_samples = int(sample_rate / fundamental_freq)
    
#     # Step 3: Identify zero crossings (rising)
#     zero_crossings = np.where(np.diff(np.sign(audio_data)) > 0)[0]
    
#     # Step 4: Extract 4 cycles of the waveform
#     for i in range(len(zero_crossings) - 4):
#         candidate_start = zero_crossings[i]
#         candidate_end = zero_crossings[i + 4]
#         if candidate_end - candidate_start >= 4 * period_samples:
#             grain = audio_data[candidate_start:candidate_end]
#             break
#     else:
#         raise ValueError("Could not find 4 complete cycles in the audio data.")
    
#     # Step 5: Resample the extracted grain to the desired wavelet length
#     resampled_grain = resample(grain, wavelet_length)
    
#     # Step 6: Generate the wavetable by scaling the amplitude of the resampled wavelet
#     wavetable = np.zeros((number_of_waves, wavelet_length))
#     for i in range(number_of_waves):
#         wavetable[i] = resampled_grain * (i / (number_of_waves - 1))
    
#     return wavetable


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

def prepare_directories(output_dir, samples_dir, wavetables_dir):
    os.makedirs(os.path.join(output_dir, samples_dir), exist_ok=True)
    os.makedirs(os.path.join(output_dir, wavetables_dir), exist_ok=True)

def compute_peak_amplitudes(wave_data, slice_ranges):
    return [np.max(np.abs(wave_data[start:end])) for start, end in slice_ranges]

def process_slices(
    wave_data, sample_rate, slice_ranges, peak_amplitudes, output_dir, samples_dir, wavetables_dir
):
    slices_info = []
    max_amplitude = max(peak_amplitudes)

    for i, (start, end) in enumerate(slice_ranges):
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
        slice_filename = os.path.join(output_dir, samples_dir, f"slice_{i}_{note}_{upper_velocity}_{lower_velocity}.wav")
        slice_data = slice_data / np.max(slice_data)
        write(slice_filename, sample_rate, (slice_data * 32767).astype(np.int16))

        # Generate and save the wavetable in the "wavetables" directory
        slice_data = change_pitch(slice_data, sample_rate, (sample_rate / 2048) * 8)
        wavetable = generate_wavetable(slice_data)
        wavetable_filename = os.path.join(output_dir, wavetables_dir, f"wavetable_{i}_{note}_{upper_velocity}_{lower_velocity}.wav")
        write(wavetable_filename, sample_rate, (wavetable * 32767).astype(np.int16))

        # Store slice information
        slice_info = SliceInfo(
            index=i + 1,
            frequency_hz=dominant_freq,
            musical_note=note,
            peak_amplitude=peak_amplitude,
            velocity=velocity,
            slice_data=slice_data,
        )
        slices_info.append(slice_info)
        print(slice_info)

    return slices_info

def split_wav(wave_data, sample_rate, num_slices, output_dir="output", samples_dir="samples", wavetables_dir="wavetables"):
    prepare_directories(output_dir, samples_dir, wavetables_dir)

    num_frames = len(wave_data)
    frames_per_slice = num_frames // num_slices
    slice_ranges = [
        (int(i * frames_per_slice), int(min((i + 1) * frames_per_slice, num_frames)))
        for i in range(num_slices)
    ]

    peak_amplitudes = compute_peak_amplitudes(wave_data, slice_ranges)
    return process_slices(wave_data, sample_rate, slice_ranges, peak_amplitudes, output_dir, samples_dir, wavetables_dir)

def split_wav_by_trans(
    wave_data, sample_rate, output_dir="output", samples_dir="samples", wavetables_dir="wavetables",
    threshold=0.05, frame_size_ms=10, sample_length=2.0
):
    prepare_directories(output_dir, samples_dir, wavetables_dir)

    split_locations = find_split_locations(wave_data, sample_rate, threshold, frame_size_ms, sample_length)
    slice_ranges = [(split_locations[i], split_locations[i + 1]) for i in range(len(split_locations) - 1)]

    peak_amplitudes = compute_peak_amplitudes(wave_data, slice_ranges)
    return process_slices(wave_data, sample_rate, slice_ranges, peak_amplitudes, output_dir, samples_dir, wavetables_dir)

# Main Function
def main():
    file_path = "CP33-EPiano1.wav"
    num_slices = 4

    # Load the .wav file
    sample_rate, wave_data = load_wav(file_path)

    # Split the wave data into slices
    # slices = split_wav(wave_data, sample_rate, num_slices)
    slices = split_wav_by_trans(
    wave_data,
    sample_rate,
    output_dir="output",
    samples_dir="samples",
    wavetables_dir="wavetables",
    threshold=0.002,
    frame_size_ms=10,
    sample_length=1.5
)

    return slices


# Example usage
if __name__ == "__main__":
    main()