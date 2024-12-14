import tkinter as tk
from tkinter import filedialog, messagebox
import os
from split_by_transient import load_wav, find_split_locations
from wave_slice import process_slices

class AudioSplitterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Splitter by Transients")

        # Variables for user input
        self.file_path = ""
        self.sensitivity = 0.5
        self.frame_size_ms = 10
        self.sample_length = 2.0  # Default minimum sample length in seconds
        self.sample_rate = None
        self.audio_data = None
        self.split_points = []

        # Frame for inputs
        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=10, pady=10)

        # File selection
        self.file_label = tk.Label(input_frame, text="No file selected", width=50, anchor="w")
        self.file_label.grid(row=0, column=0, columnspan=2)
        self.select_button = tk.Button(input_frame, text="Select File", command=self.select_file)
        self.select_button.grid(row=0, column=2)

        # Frame size input
        self.frame_size_label = tk.Label(input_frame, text="Frame Size (ms):")
        self.frame_size_label.grid(row=1, column=0, sticky="e")
        self.frame_size_entry = tk.Entry(input_frame)
        self.frame_size_entry.insert(tk.END, str(self.frame_size_ms))
        self.frame_size_entry.grid(row=1, column=1)

        # Frame size note
        self.frame_size_note = tk.Label(input_frame, text="<10ms may detect low frequencies as transients.", font=("Arial", 8), fg="gray")
        self.frame_size_note.grid(row=2, column=0, columnspan=3, pady=5, sticky="w")              

        # Sensitivity input
        self.sensitivity_label = tk.Label(input_frame, text="Sensitivity:")
        self.sensitivity_label.grid(row=3, column=0, sticky="e")
        self.sensitivity_entry = tk.Entry(input_frame)
        self.sensitivity_entry.insert(tk.END, str(self.sensitivity))
        self.sensitivity_entry.grid(row=3, column=1)

        # Sensitivity note
        self.sensitivity_note = tk.Label(input_frame, text="0.99 is very sensitive, 0.01 will only detect very large transients", font=("Arial", 8), fg="gray")
        self.sensitivity_note.grid(row=4, column=0, columnspan=3, pady=5, sticky="w")

        # Minimum Sample Length input
        self.sample_length_label = tk.Label(input_frame, text="Minimum Sample Length (s):")
        self.sample_length_label.grid(row=5, column=0, sticky="e")
        self.sample_length_entry = tk.Entry(input_frame)
        self.sample_length_entry.insert(tk.END, str(self.sample_length))
        self.sample_length_entry.grid(row=5, column=1)

        # Minimum Sample Length note
        self.sample_length_note = tk.Label(input_frame, text="Shorter values may create too many small chunks.", font=("Arial", 8), fg="gray")
        self.sample_length_note.grid(row=6, column=0, columnspan=3, pady=5, sticky="w")

        # Preview and Slice buttons
        self.preview_button = tk.Button(self.root, text="Preview", command=self.preview)
        self.preview_button.pack(pady=5)
        self.slice_button = tk.Button(self.root, text="Slice and Save", command=self.slice_and_save)
        self.slice_button.pack(pady=5)

    def select_file(self):
        # Open a file dialog to select the WAV file
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=os.path.basename(file_path))

    def preview(self):
        # Get user inputs
        try:
            self.frame_size_ms = int(self.frame_size_entry.get())
            self.sensitivity = float(self.sensitivity_entry.get())
            self.sample_length = float(self.sample_length_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid frame size, sensitivity, or sample length value.")
            return

        if not self.file_path:
            messagebox.showerror("File Error", "Please select a WAV file.")
            return

        # Load the WAV file
        self.sample_rate, self.audio_data = load_wav(self.file_path)

        # Find split locations
        self.split_points = find_split_locations(self.audio_data, self.sample_rate,
                                                 sensitivity=self.sensitivity, frame_size_ms=self.frame_size_ms,
                                                 sample_length=self.sample_length, plot=True)

    def slice_and_save(self):
        if not self.file_path:
            messagebox.showerror("File Error", "Please select a WAV file.")
            return

        if not self.split_points:
            messagebox.showerror("Preview Error", "Please preview the splits first.")
            return

        # Get user inputs
        try:
            output_folder = filedialog.askdirectory(title="Select Output Directory")
            if not output_folder:
                return  # User canceled the directory selection
        except Exception as e:
            messagebox.showerror("Directory Error", f"Error selecting output directory: {e}")
            return

        # Split and save audio chunks
        process_slices(wave_data=self.audio_data, sample_rate=self.sample_rate, slice_ranges=self.split_points, output_dir=output_folder)
        messagebox.showinfo("Success", f"Audio has been split and saved in {output_folder}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioSplitterApp(root)
    root.mainloop()
