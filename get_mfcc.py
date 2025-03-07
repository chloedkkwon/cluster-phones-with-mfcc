# Written by Chloe D. Kwon (dk837@cornell.edu)
# March 4, 2025
# How to run: python get_mfcc.py ../_data/wav ../_data

import os
import sys
import numpy as np
import librosa


def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
    # Extracts MFCCs from an audio file
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfccs
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_dir(audio_dir, data_dir):
    # Processes all WAV files and saves MFCCs as .npy files
    if not os.path.isdir(audio_dir):
        print(f"Error: {audio_dir} is not a valid directory")
        return

    output_dir = os.path.join(data_dir, "mfcc_features")
    os.makedirs(output_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    if not wav_files:
        print(f"No WAV files found in {audio_dir}")
        return

    for wav_file in wav_files:
        audio_path = os.path.join(audio_dir, wav_file)
        mfccs = extract_mfcc(audio_path)
        if mfccs is not None:
            mfcc_filename = os.path.join(output_dir, wav_file.replace(".wav", ".npy"))
            np.save(mfcc_filename, mfccs)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python get_mfcc.py <wav_directory> <mfcc_directory>")
        sys.exit(1)

    input1_audio = sys.argv[1]
    input2_data = sys.argv[2]
    process_dir(input1_audio, input2_data)

