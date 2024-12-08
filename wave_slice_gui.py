import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from wave_slice import split_wav  # Import your function

class WaveSlicerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wave Slicer")

        # Initialize variables
        self.audio_data = None
        self.sample_rate = None
        self.num_slices = 0

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Button to load the audio file
        self.load_button = tk.Button(self.root, text="Load Audio File", command=self.load_audio)
        self.load_button.pack(pady=10)

        # Entry field for the number of slices
        self.num_slices_label = tk.Label(self.root, text="Number of Slices:")
        self.num_slices_label.pack(pady=5)
        self.num_slices_entry = tk.Entry(self.root)
        self.num_slices_entry.pack(pady=5)
        self.num_slices_entry.bind('<KeyRelease>', self.on_slices_entry_change)

        # Buttons to shift the waveform
        self.shift_label = tk.Label(self.root, text="Shift Samples:")
        self.shift_label.pack(pady=5)

        # Create shift buttons
        self.shift_buttons_frame = tk.Frame(self.root)
        self.shift_buttons_frame.pack(pady=10)

        self.shift_minus_1000 = tk.Button(self.shift_buttons_frame, text="-1000", command=lambda: self.shift_waveform(-1000))
        self.shift_minus_1000.grid(row=0, column=0, padx=5)

        self.shift_minus_100 = tk.Button(self.shift_buttons_frame, text="-100", command=lambda: self.shift_waveform(-100))
        self.shift_minus_100.grid(row=0, column=1, padx=5)

        self.shift_plus_100 = tk.Button(self.shift_buttons_frame, text="+100", command=lambda: self.shift_waveform(100))
        self.shift_plus_100.grid(row=0, column=2, padx=5)

        self.shift_plus_1000 = tk.Button(self.shift_buttons_frame, text="+1000", command=lambda: self.shift_waveform(1000))
        self.shift_plus_1000.grid(row=0, column=3, padx=5)

        # Button to slice and save the wave
        self.slice_button = tk.Button(self.root, text="Slice and Save", command=self.save_slices)
        self.slice_button.pack(pady=10)

        # Create the canvas for displaying the waveform and slices
        self.fig, (self.ax, self.ax_first, self.ax_last) = plt.subplots(3, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # A tk.DrawingArea
        self.canvas.get_tk_widget().pack(pady=20)

    def load_audio(self):
        # Open file dialog to select audio file
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        
        if file_path:
            # Load the audio file using librosa
            self.sample_rate, self.audio_data = wavfile.read(file_path)
            # Convert to mono if it's stereo

            # Normalize data if needed
            if self.audio_data.dtype == np.int16:
                self.audio_data = (self.audio_data / 32768.0).astype(np.float32)
            elif self.audio_data.dtype == np.int32:
                self.audio_data = (self.audio_data / 2147483648.0).astype(np.float32)
            elif self.audio_data.dtype == np.uint8:
                self.audio_data = ((self.audio_data - 128) / 128.0).astype(np.float32)

            if len(self.audio_data.shape) > 1:  # Check if the audio is stereo
                self.audio_data = self.audio_data.mean(axis=1).astype(self.audio_data.dtype)

            # Display the waveform on the canvas
            self.plot_waveform()

    def plot_waveform(self):
        # Clear previous plots
        self.ax.clear()
        self.ax_first.clear()
        self.ax_last.clear()

        # Plot the audio waveform
        self.ax.plot(np.linspace(0, len(self.audio_data) / self.sample_rate, len(self.audio_data)), self.audio_data)
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Waveform')
        self.canvas.draw()

    def shift_waveform(self, shift_value):
        if self.audio_data is None:
            return

        # Ensure the shift value does not exceed audio length
        shift_value = int(shift_value)
        if abs(shift_value) > len(self.audio_data):
            shift_value = len(self.audio_data) if shift_value > 0 else -len(self.audio_data)

        # Apply the shift
        if shift_value > 0:
            # Add zeros at the start and remove samples from the end
            self.audio_data = np.concatenate((np.zeros(shift_value), self.audio_data))[:len(self.audio_data)]
        elif shift_value < 0:
            # Remove samples from the start and add zeros at the end
            self.audio_data = np.concatenate((self.audio_data[-shift_value:], np.zeros(-shift_value)))

        # Replot the waveform and update slices
        self.plot_waveform()
        self.slice_wave()

    def on_slices_entry_change(self, event):
        self.slice_wave()

    def slice_wave(self):
        try:
            self.num_slices = int(self.num_slices_entry.get())
        except ValueError:
            return

        if self.audio_data is not None and self.num_slices > 0:
            slice_length = len(self.audio_data) // self.num_slices
            slice_indices = np.linspace(0, len(self.audio_data), self.num_slices + 1, dtype=int)

            self.plot_waveform()

            for idx in slice_indices:
                self.ax.axvline(x=idx / self.sample_rate, color='r', linestyle='--', lw=1)

            first_slice = self.audio_data[slice_indices[0]:slice_indices[1]]
            time_first = np.linspace(0, len(first_slice) / self.sample_rate, len(first_slice))
            self.ax_first.plot(time_first, first_slice, color='g')
            self.ax_first.set_title('First Slice')

            last_slice = self.audio_data[slice_indices[-2]:slice_indices[-1]]
            time_last = np.linspace(0, len(last_slice) / self.sample_rate, len(last_slice))
            self.ax_last.plot(time_last, last_slice, color='b')
            self.ax_last.set_title('Last Slice')

            self.canvas.draw()

    def save_slices(self):
        if self.audio_data is None or self.num_slices <= 0:
            return

        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return

        split_wav(self.audio_data, self.sample_rate, self.num_slices, output_dir)
        tk.messagebox.showinfo("Save Slices", f"Slices saved successfully in {output_dir}")

# Create the Tkinter window
root = tk.Tk()
app = WaveSlicerApp(root)
root.mainloop()
