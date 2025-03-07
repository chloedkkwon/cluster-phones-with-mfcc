# Written by Chloe D. Kwon (dk837@cornell.edu)
# March 4, 2025
# How to run: python get_mfcc_phone.py ../_data/wav ../_data
# MFCC in each audio file -> segment these feature vectors corresponding to each phoneme

# Input: mfcc files, textgrid files
# Output:label and feature array

import os
import sys
import numpy as np
import pandas as pd
import tgt  # For parsing TextGrid files
import pickle

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python get_mfcc_phone.py <mfcc_directory> <textgrid_directory>")
        sys.exit(1)

    feature_dir = sys.argv[1]
    textgrid_dir = sys.argv[2]


''' Get list of feature files '''
feature_files = [f for f in os.listdir(feature_dir) if f.endswith(".npy")]
print(f"Found {len(feature_files)} feature files.")


''' Load features and filenames '''
data = []
for feature_file in feature_files:
    # Load feature vector
    feature_path = os.path.join(feature_dir, feature_file)
    feature_vec = np.load(feature_path)

    # Find corresponding TextGrid
    base_name = os.path.splitext(feature_file)[0]
    textgrid_path = os.path.join(textgrid_dir, base_name + ".TextGrid")

    if not os.path.exists(textgrid_path):
        print(f"Warning: No TextGrid found for {base_name}")
        continue  # Skip this file if no TextGrid is found

    # Load TextGrid and extract segment label
    tg = tgt.io.read_textgrid(textgrid_path)
    phone_tier = tg.tiers[1]  # Tier 0 is word, tier 1 is phone
    word_tier = tg.tiers[0]

    # Split feature vector into segments based on number of time frames
    num_time_frames = feature_vec.shape[1]
    start_time = phone_tier.start_time
    end_time = phone_tier.end_time
    time_per_frame = (end_time - start_time) / num_time_frames # time step per frame

    # Iterate over each time step in the feature vector
    for i in range(num_time_frames):
        current_time = start_time + i * time_per_frame

        phonemes = [interval.text for interval in phone_tier.intervals if interval.start_time <= current_time <= interval.end_time]
        words = [interval.text for interval in word_tier.intervals if interval.start_time <= current_time <= interval.end_time]

        # Store the feature, corresponding phonemes, and words
        data.append({
            "file": base_name,
            "time_frame": i,
            "feature": feature_vec[:,i],
            "phonemes": ",".join(phonemes) if phonemes else "NA",
            "words": ",".join(words) if words else "NA"
        })

df = pd.DataFrame(data)
print(df.head())

# Save as CSV for future use
df.to_csv("features_with_annotations.csv", index=False)

# Group by phonemes and words & stack features
ph_data = {}
for (file, phoneme), group in df.groupby(["file", "phonemes"]):
    feat = np.vstack(group["feature"].values)
    if phoneme not in ph_data:
        ph_data[phoneme] = []
    ph_data[phoneme].append(feat)


''' Save '''
with open('ph_data.pickle', 'wb') as hdl:
    pickle.dump(ph_data, hdl)