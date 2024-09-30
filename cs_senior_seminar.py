from music21 import converter
import numpy as np
import json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def abc_to_midi(input_file: str, output_file: str):
    with open(input_file, 'r') as file:
        abc_content = file.read()
    
    abc_to_midi = converter.parse(abc_content)
    
    abc_to_midi.write('midi', fp=output_file)
    print(f"MIDI file saved to {output_file}")

    return abc_to_midi

def extract_features(midi_tunes):
    features = []
    for midi_tune in midi_tunes:
        feature = extract_feature(midi_tune)
        features.append(feature)
    return features

def extract_feature(midi_format):
    pitches = []
    durations = []
    rests = 0
    chords = 0
    intervals = []
    notes = 0

    for element in midi_format.flat.notes:
        notes += 1
        if element.isRest:
            rests += 1
        elif element.isChord:
            chords += 1
        else:
            pitches.append(element.pitch.midi)
            durations.append(element.duration.quarterLength)
            intervals.append(element.pitch.ps)

    avg_pitch = sum(pitches) / len(pitches) if pitches else 0
    pitch_range = max(pitches) - min(pitches) if pitches else 0
    pitch_sd = np.std(pitches) if pitches else 0
    pitches_len = len(pitches) if pitches else 0
    
    avg_duration = sum(durations) / len(durations) if durations else 0
    duration_range = max(durations) - min(durations) if durations else 0
    duration_sd = np.std(durations) if durations else 0
    total_duration = sum(durations) if durations else 0

    avg_interval = sum(intervals) / len(intervals) if intervals else 0
    interval_range = max(intervals) - min(intervals) if intervals else 0
    interval_sd = np.std(intervals) if intervals else 0

    features = {
        'notes': notes,
        'rests': rests,
        'chords': chords,
        'avg_pitch': avg_pitch,
        'pitch_range': pitch_range,
        'pitch_sd': pitch_sd,
        'pitches_len': pitches_len,
        'avg_duration': avg_duration,
        'duration_range': duration_range,
        'duration_sd': duration_sd,
        'total_duration': total_duration,
        'avg_interval': avg_interval,
        'interval_range': interval_range,
        'interval_sd': interval_sd
    }

    return features

def k_means_clustering(dataset, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(dataset)
    labels = kmeans.labels_

    return labels

def create_dataset(features):
    print(len(features))
    dataset = []
    for feature in features:
        dataset.append([
            feature['notes'],
            feature['rests'],
            feature['chords'],
            feature['avg_pitch'],
            feature['pitch_range'],
            feature['pitch_sd'],
            feature['pitches_len'],
            feature['avg_duration'],
            feature['duration_range'],
            feature['duration_sd'],
            feature['total_duration'],
            feature['avg_interval'],
            feature['interval_range'],
            feature['interval_sd']
        ])
    dataset = np.array(dataset)
    return dataset

def read_abcs(file):
    with open(file, 'r') as f:
        abc_tunes = f.read().split('\n\n')
    return abc_tunes

def convert_abc_to_midi(abc_tunes):
    midi_tunes = []
    for abc_tune in abc_tunes:
        midi = converter.parse(abc_tune)
        midi_tunes.append(midi)
    return midi_tunes

def main():
    input_file = '/Users/maheen/Desktop/cs senior seminar/sample_abc.txt'
    output_file = '/Users/maheen/Desktop/cs senior seminar/result.txt'

    abc_tunes = read_abcs(input_file)
    #print(abc_tunes)
    midi_tunes = convert_abc_to_midi(abc_tunes)
    print(len(midi_tunes))

    # midi = abc_to_midi(input_file, output_file)
    # #print(midi)

    features = extract_features(midi_tunes)
    # print(json.dumps(features, indent=4))
    dataset = create_dataset(features)
    # print(dataset)

    labels = k_means_clustering(dataset, 2)
    print(labels)



if __name__ == '__main__':
    main()