import numpy as np
import os
from music21 import note, chord, stream, converter, instrument
from tensorflow.keras.models import load_model

# ---------- Load notes from MIDI files ----------
notes = []
midi_folder = "data/midi_files"

for file in os.listdir(midi_folder):
    if file.endswith(".mid"):
        midi = converter.parse(os.path.join(midi_folder, file))
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

# ---------- Prepare input sequences ----------
sequence_length = 50
pitchnames = sorted(set(notes))
note_to_int = {note: number for number, note in enumerate(pitchnames)}

network_input = []

for i in range(len(notes) - sequence_length):
    sequence = notes[i:i + sequence_length]
    network_input.append([note_to_int[n] for n in sequence])

network_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
network_input = network_input / float(len(pitchnames))

# ---------- Load trained model ----------
model = load_model("music_model.h5")

# ---------- Generate new music ----------
start = np.random.randint(0, len(network_input) - 1)
pattern = network_input[start]

prediction_output = []

for _ in range(200):
    prediction = model.predict(pattern.reshape(1, pattern.shape[0], 1), verbose=0)
    index = np.argmax(prediction)
    result = pitchnames[index]
    prediction_output.append(result)

    pattern = np.append(pattern, [[index / float(len(pitchnames))]], axis=0)
    pattern = pattern[1:]

# ---------- Convert output to MIDI ----------
output_notes = []
offset = 0

for pattern in prediction_output:
    if '.' in pattern:
        notes_in_chord = pattern.split('.')
        chord_notes = [note.Note(int(n)) for n in notes_in_chord]
        new_chord = chord.Chord(chord_notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='generated_music.mid')

print("Music generated successfully: generated_music.mid")
