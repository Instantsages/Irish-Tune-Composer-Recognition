from sklearn.cluster import KMeans
from music21 import converter
import numpy as np

def extract_tempo(abc_notation):
    return 10

def extract_duration(abc_notation):
    return 300

def extract_key_signature(abc_notation):
    return 200

def processing_pipeline(tunes_tuples):
    tunes_extracted_features = {}
    for tune_name, tune_composer, abc_notation in tunes_tuples:
        midi = converter.parse(abc_notation)
        features = extract_features(midi)
        tunes_extracted_features[tune_name] = features
        tunes_extracted_features[tune_name]['composer'] = tune_composer
        
    return tunes_extracted_features

def extract_features(midi_format):
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