from postprocessing import *
from preprocessing import *

def main():
    data_path = './dataset/midiclassic/'
    midi_files = []

    for foldername, _, filesnames in os.walk(data_path):
        for filename in filesnames:
            if filename.lower().endswith('.mid'):
                file_path = os.path.join(foldername, filename)
                midi_files.append(file_path)

    print(f"{len(midi_files)} MIDI files found to preprocess!")

    tensor_folder = './dataset/midiclassic_tensor/'
    os.makedirs(tensor_folder, exist_ok=True)

    idx = 0
    for i, midi_file in enumerate(midi_files):
        try:
            if idx % 10 == 0:
                print(f"\rProcessing file: {i+1}/{len(midi_files)} ({(i+1)/len(midi_files)*100:.2f}%) [{midi_file}]", end='', flush=True)
            
            pitch_tensor, _ = parse_midi_to_tensor(midi_file)
            pitch_tensor = pitch_tensor.reshape(-1, pitch_tensor.shape[-1])

            #convert pitch tensor numpy array to dtype int between 0 and 127;
            np.save(tensor_folder + f"i{idx}_res{RESOLUTION}.npy", pitch_tensor)
            idx += 1
        except Exception as e:
            print(f"\nError processing file {midi_file}: {e}")


if __name__ == "__main__":
    main()