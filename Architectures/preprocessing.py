import mido
import os
import numpy as np

NBR_CHANNELS = 16
NBR_NOTES = 128
RESOLUTION = 96

def parse_midi_to_tensor(file_path):
    
    midi_file = mido.MidiFile(file_path)

    ticks_per_beat = midi_file.ticks_per_beat
    
    pitch_array = [[[] for _ in range(NBR_NOTES)] for _ in range(NBR_CHANNELS)]

    max_time = 0

    for track in midi_file.tracks:
        memory_time = 0
        for msg in track:
            memory_time += int((msg.time/ticks_per_beat)*RESOLUTION)
            if msg.type == 'note_on' or msg.type == 'note_off':
                channel = msg.channel
                velocity = msg.velocity if msg.type == 'note_on' else 0
                note = msg.note

                pitch_array[channel][note].append([memory_time, velocity])

        max_time = max(max_time, memory_time)

    pitch_tensor = np.zeros((16, 128, max_time), dtype=np.uint8)

    for c in range(len(pitch_array)):
        memory = np.zeros((128, 2))

        for n in range(len(pitch_array[c])):
            if len(pitch_array[c][n]) == 0:
                continue
            
            for e in pitch_array[c][n]:
                event_absolute_index = e[0]
                event_value = e[1]

                pitch_tensor[c, n, int(memory[n, 0]):int(event_absolute_index + memory[n, 0])] = memory[n, 1]

                memory[n,0] = event_absolute_index
                memory[n,1] = event_value
            
            pitch_tensor[c, n, int(memory[n, 0]):] = memory[n, 1]

    return pitch_tensor, ticks_per_beat