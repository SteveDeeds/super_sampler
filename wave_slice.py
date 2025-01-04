import os
import numpy as np
from scipy.io.wavfile import write
from scipy.fft import fft, ifft
from SliceInfo import SliceInfo, frequency_to_note, analyze_frequency, calculate_velocity, change_pitch, tune_to_nearest_note, load_wav, prepare_directories

def generate_wavetable(audio_data, wavelet_length=2048, number_of_waves=256, sample_rate=44100, threshold =0):
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

    phase = []
    # Generate random phases for the first half
    first_half = []
    second_half_rev = []
    for _ in range(wavelet_length // 2):
        angle = np.random.uniform(0, 2 * np.pi)  # Random angle between 0 and 2*pi radians
        z = np.cos(angle) + 1j * np.sin(angle)  # Complex number representation
        first_half.append(z)
        z = np.cos(angle+np.pi) + 1j * np.sin(angle+np.pi)  # Complex number representation
        second_half_rev.append(z)

    second_half = second_half_rev[::-1]
    phase = first_half + second_half

    # # zero phase
    # phase = np.ones(wavelet_length)* -1j
    # phase[len(phase)//2:] = 1j

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

        # Set values below threshold_level to zero
        threshold_level = np.max(mag_fft)*threshold
        mag_fft[mag_fft < threshold_level] = 0

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

    #normalize wavetable
    full_wavetable = full_wavetable/np.max(np.abs(full_wavetable))

    return full_wavetable

def process_slices(
    wave_data, sample_rate, slice_ranges, output_dir):
    print(slice_ranges)
    peak_amplitudes = [np.max(np.abs(wave_data[start:end])) for start, end in slice_ranges]
    slices_info = []
    max_amplitude = max(peak_amplitudes)

    for i, (start, end) in enumerate(slice_ranges):
        slice_data = wave_data[start:end]

        # Tune the note to the nearest pitch
        slice_data = tune_to_nearest_note(slice_data, sample_rate)

        # Analyze frequency, determine musical note, and calculate velocity
        dominant_freq = analyze_frequency(slice_data, sample_rate)
        note, note_number = frequency_to_note(dominant_freq)
        peak_amplitude = peak_amplitudes[i]
        velocity = calculate_velocity(peak_amplitude, max_amplitude)

        # Save the slice in the "samples" directory
        upper_velocity = min(127, velocity + 10)
        lower_velocity = max(0, velocity - 10)
        slice_filename = os.path.join(output_dir, "samples", f"slice_{i+1:03d}_{note}_{note_number}_{upper_velocity}_{lower_velocity}.wav")
        slice_data = slice_data / np.max(np.abs(slice_data))
        prepare_directories(output_dir=output_dir)
        write(slice_filename, sample_rate, (slice_data * 32767).astype(np.int16))

        # Generate and save the wavetable in the "wavetables" directory
        slice_data = change_pitch(slice_data, sample_rate, (sample_rate / 2048) * 8)
        wavetable = generate_wavetable(slice_data, sample_rate=sample_rate)
        wavetable_filename = os.path.join(output_dir, "wavetables", f"wavetable_{i+1:03d}_{note}_{note_number}_{upper_velocity}_{lower_velocity}.wav")
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

def split_wav(wave_data, sample_rate, num_slices, output_dir="output"):
    prepare_directories(output_dir)

    num_frames = len(wave_data)
    frames_per_slice = num_frames // num_slices
    slice_ranges = [
        (int(i * frames_per_slice), int(min((i + 1) * frames_per_slice, num_frames)))
        for i in range(num_slices)
    ]

    return process_slices(wave_data, sample_rate, slice_ranges, output_dir)

# Main Function
def main():
    file_path = "CP33-EPiano1.wav"
    num_slices = 30

    # Load the .wav file
    sample_rate, wave_data = load_wav(file_path)

    # Split the wave data into slices
    slices = split_wav(wave_data, sample_rate, num_slices)

    return slices


# Example usage
if __name__ == "__main__":
    main()