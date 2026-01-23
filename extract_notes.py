from music21 import converter, instrument, note, chord
import os

notes = []

midi_folder = "data/midi_files"

print("Reading MIDI files...")

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
            print("Skipped (invalid):", file)

print("DONE")
print("Total notes extracted:", len(notes))
