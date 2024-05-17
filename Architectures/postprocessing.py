import mido
import os
import random
import numpy as np

NBR_CHANNELS = 16
NBR_NOTES = 128
RESOLUTION = 96

def tensor_to_midi(pitch_tensor):

    pitch_array = [[] for _ in range(NBR_CHANNELS)]
    
    for c in range(NBR_CHANNELS):
        memory_pitch = np.zeros((128))
        memory_pitch_index = 0
        
        if np.all(pitch_tensor[c, :, :] == 0):
            continue

        for i in range(pitch_tensor.shape[-1]):
            if np.all((memory_pitch[:] - pitch_tensor[c, :, i]) == 0):
                continue

            event_index = np.argwhere(pitch_tensor[c, :, i] - memory_pitch[:] != 0).flatten()

            for note in event_index:
                delta_i = i - memory_pitch_index
                delta_ticks = delta_i
                pitch_array[c].append([delta_ticks, note, pitch_tensor[c, note, i]])

                memory_pitch_index = i
                memory_pitch[note] = pitch_tensor[c, note, i]

    midi_file = mido.MidiFile(ticks_per_beat=RESOLUTION)
    
    for c in range(NBR_CHANNELS):
        if len(pitch_array[c]) == 0:
            continue
        
        track = mido.MidiTrack()
        for e in pitch_array[c]:
            velocity = int(e[2])
            note = int(e[1])
            ticks = int(e[0])

            track.append(mido.Message('note_on' if velocity > 0 else 'note_off', 
                                              note=note, velocity=velocity, 
                                              time=ticks, channel=c))
        midi_file.tracks.append(track)

    return midi_file