import numpy as np
import os
from music21 import converter, instrument, note, chord
from tensorflow.keras.utils import to_categorical
from model import create_model

notes = []
midi_folder = "data/midi_files"

print("Loading MIDI files...")

# ---------- STEP 1: Read MIDI files ----------
for file in os.listdir(midi_folder):
    if file.endswith(".mid"):
        file_path = os.path.join(midi_folder, file)
        try:
            midi = converter.parse(file_path)
            parts = instrument.partitionByInstrument(midi)

            if parts:
                elements = parts.parts[0].recurse()
            else:
                elements = midi.flat.notes

            for element in elements:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

            print("Processed:", file)

        except:
            print("Skipped:", file)

print("Total notes:", len(notes))

# ---------- STEP 2: Prepare sequences ----------
sequence_length = 50
pitchnames = sorted(set(notes))

note_to_int = {note: number for number, note in enumerate(pitchnames)}

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]

    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)

network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(len(pitchnames))
network_output = to_categorical(network_output)

print("Training patterns:", n_patterns)

# ---------- STEP 3: Create and train model ----------
model = create_model(
    input_shape=(network_input.shape[1], network_input.shape[2]),
    output_units=len(pitchnames)
)

model.fit(
    network_input,
    network_output,
    epochs=20,
    batch_size=64
)

model.save("music_model.h5")
print("Model training complete. Saved as music_model.h5")
