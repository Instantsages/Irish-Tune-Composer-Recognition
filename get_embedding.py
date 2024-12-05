"""
    File:        get_data.py
    Author:      Yifan Wu
    Project:     CS 330 - Irish Tune Composer Recognition
    Semester:    Fall 2024
    Description: Get embeddings from .bin file
"""

import re
import numpy as np

def get_embeddings(input_file = "embeddings.bin", prefix = "Users-wu_ivan-Desktop-Ivan-CS_Senior_Project-midi2vec-midi2edgelist-MIDI-"):
    all_combos = []
    with open(input_file, 'r') as file:
        for line in file:
            if line.startswith(prefix):
                all_combos.append(line.split(' '))

    composers = []
    embeddings = []

    for combo in all_combos:
        info = combo[0].removeprefix(prefix)
        current_composer = re.sub(r"_\d+$", "", info).replace('_', ' ')
        current_embeddings = combo[1:]
        current_embeddings = [float(data.strip()) for data in current_embeddings]

        composers.append(current_composer)
        embeddings.append(current_embeddings)

    return np.array(embeddings), composers
        



def main():
    embeddings, composers = get_embeddings()
    print(embeddings)
    

if __name__ == "__main__":
    main()