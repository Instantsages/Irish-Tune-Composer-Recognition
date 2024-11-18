from music21 import converter, chord, interval, note, stream
import numpy as np
import json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


"""Global Variable; list of names for features that need to be extracted"""
FEATURES = [  #'notes',
    #'rests',
    #'chords',
    "avg_pitch",
    "pitch_range",
    "pitch_sd",
    #'pitches_len',
    "avg_duration",
    "duration_range",
    "duration_sd",
    #'total_duration',
    "avg_interval",
    "interval_range",
    "interval_sd",
    "contour_up",
    "contour_down",
    "note_density",
    "syncopation_ratio",
    "different_rhythms",
    "different_rhythms_ratio",
]


def abc_to_midi(input_file: str, output_file: str):
    with open(input_file, "r") as file:
        abc_content = file.read()

    abc_to_midi = converter.parse(abc_content)

    abc_to_midi.write("midi", fp=output_file)
    print(f"MIDI file saved to {output_file}")

    return abc_to_midi


def extract_features(midi_tunes, feature_names=FEATURES):
    """takes in a dictionary of composer to list of midi tunes"""
    features = {}
    for composer, midi_tunes in midi_tunes.items():
        for midi_tune in midi_tunes:
            feature = extract_feature(midi_tune, feature_names)
            if composer not in features:
                features[composer] = [feature]
            else:
                features[composer].append(feature)
    return features


def extract_feature(midi_format, feature_names=FEATURES):
    pitches = []
    durations = []
    rests = 0
    chords = 0
    intervals = []
    interval_cents = []
    notes = 0
    syncopation = 0
    melodic_contour = []
    rhytimic_features = []

    for element in midi_format.flat.notes:
        notes += 1
        if element.isRest:
            rests += 1
        elif element.isChord:
            chords += 1
        else:
            pitches.append(element.pitch)
            durations.append(element.duration.quarterLength)
            intervals.append(element.pitch)

            if len(pitches) > 1:
                intvl = interval.Interval(noteStart=pitches[-2], noteEnd=pitches[-1])
                interval_cents.append(intvl.cents)

            # whether pitch goes up, down, or stays the same
            if len(pitches) > 1:
                if pitches[-1] > pitches[-2]:
                    melodic_contour.append(1)  # up
                elif pitches[-1] < pitches[-2]:
                    melodic_contour.append(-1)  # down
                else:
                    melodic_contour.append(0)  # same

            if element.offset % 1 != 0:
                syncopation += 1

            rhytimic_features.append(element.duration.quarterLength)

    pitches = [p.ps for p in pitches]
    intervals = [i.ps for i in intervals]

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

    contour_up = (
        melodic_contour.count(1) / len(melodic_contour) if melodic_contour else 0
    )
    contour_down = (
        melodic_contour.count(-1) / len(melodic_contour) if melodic_contour else 0
    )

    note_density = notes / total_duration if total_duration > 0 else 0
    syncopation_ratio = syncopation / notes if notes > 0 else 0

    different_rhythms = len(set(rhytimic_features))
    different_rhythms_ratio = (
        different_rhythms / len(rhytimic_features) if rhytimic_features else 0
    )

    features = {}

    for feature_name in feature_names:
        features[feature_name] = eval(feature_name)

    return features


def k_means_clustering(dataset, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(dataset)
    labels = kmeans.labels_

    return labels


def create_dataset(features):
    """takes in a dictionary of composer to list of features, which is each a dictionary of features to values"""
    # print(len(features))
    dataset = []
    composers = []
    for composer, features in features.items():
        for feature in features:
            dataset.append([feature[feature_name] for feature_name in feature])
            composers.append(composer)
    dataset = np.array(dataset)
    return dataset, composers


def read_abcs(file):
    abc_tunes_with_composer = {}
    with open(file, "r") as f:
        abc_tunes = f.read().split("\n\n")
    for abc_tune in abc_tunes:
        abc_tune = abc_tune.split("\n")
        composer = abc_tune[0].split(":")[1].strip()
        if composer not in abc_tunes_with_composer:
            abc_tunes_with_composer[composer] = ["\n".join(abc_tune[1:])]
        else:
            abc_tunes_with_composer[composer].append("\n".join(abc_tune[1:]))

    # for key, value in abc_tunes_with_composer.items():
    #     print("KEY: ", key, "VALUES", value)

    return abc_tunes_with_composer


def convert_abc_to_midi(abc_tunes):
    midi_tunes = {}
    for composer, abc_tunes in abc_tunes.items():
        for abc_tune in abc_tunes:
            midi = converter.parse(abc_tune)
            if composer not in midi_tunes:
                midi_tunes[composer] = [midi]
            else:
                midi_tunes[composer].append(midi)
    return midi_tunes


def visualize_clusters(dataset, labels, composers):
    # Reduce dimensions with PCA for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(dataset)

    # Create a scatter plot
    plt.figure()

    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_mask = labels == label
        plt.scatter(
            reduced_data[label_mask, 0],
            reduced_data[label_mask, 1],
            label=f"Cluster {label}",
        )

    # Add composer annotations
    for i, composer in enumerate(composers):
        plt.annotate(composer, (reduced_data[i, 0], reduced_data[i, 1]))

    plt.title("K-Means Clustering of Composers")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()


def main():
    input_file = "abc.txt"
    output_file = "result.txt"

    abc_tunes = read_abcs(input_file)

    # print(abc_tunes)
    midi_tunes = convert_abc_to_midi(abc_tunes)
    # print(len(midi_tunes))

    # midi = abc_to_midi(input_file, output_file)
    # #print(midi)

    features = extract_features(midi_tunes)
    # print(json.dumps(features, indent=4))
    dataset, composers = create_dataset(features)
    print(dataset)

    num_composers = 8
    labels = k_means_clustering(dataset, num_composers)

    visualize_clusters(dataset, labels, composers)


if __name__ == "__main__":
    main()
