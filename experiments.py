"""
    File:        experiments.py
    Author:      Ahmed Abdelrehim, Maheen Masoud, and Yifan Wu
    Project:     CS 330 - Irish Tune Composer Recognition
    Semester:    Fall 2024
    Description: Conducts experiments for k-means clustering
"""

from cs_senior_seminar import * 

from itertools import chain, combinations

from sklearn.metrics import adjusted_rand_score

import heapq

import warnings
        
# CITE: https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
# DESC: Source of this function for computing power set
def powerset_list(s):
    """
    Givens a list, returns its power set as a list
    powerset([1,2,3]) --> [() (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)]

    Keyword arguments:
        s (list): A given list
    
    Returns (list):
        The power set of s, with each subset as a tuple
    """
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))[1:]


def experiment_features(input_file = 'sample_abc.txt', output_file = 'results.txt', first_k = 30):
    """
    Experiments with all possible combinations of features, outputing the best combinations with corresponding abc rand index

    Keyword arguments:
        input_file (string): Name of the file which contains the abc notations of tunes
        output_file (string): Name of the file to write results to
        first_k (int): The number of best combinations to store and output   

    """
    abc_tunes = read_abcs(input_file)
    midi_tunes = convert_abc_to_midi(abc_tunes)

    # Finding all possible combinations of available features
    global FEATURES
    power_features = powerset_list(FEATURES)

    assert len(power_features) > first_k

    # Priority queue based on adjusted rand index of a combination's grouping, 
    result_heap = []
    # Keep record of the item last popped to reduce practical running time
    last_popped = float('-inf')

    for i, current_set in enumerate(power_features):
        # For each combo of features, extract the features, cluster, and calculate adjusted rand index
        features = extract_features(midi_tunes, list(current_set))

        dataset, composers = create_dataset(features)

        num_composers = 3
        labels = k_means_clustering(dataset, num_composers)

        rand_index = adjusted_rand_score(composers, labels)

        # Enqueuing is only necessary when rand index is greater than the last popped combo
        if rand_index > last_popped:
            print(f"Tested subset {i} with rand index {rand_index}; pushed to heap")
            heapq.heappush(result_heap, (rand_index, current_set))
        if len(result_heap) > first_k:
            last_popped, _ = heapq.heappop(result_heap)
        
    result_heap.sort()

    # Write to output file
    with open(output_file,"w") as file:
        for item in result_heap:
            file.write(f"{item}\n")
        

        
# Controls operation of experiments
def main():
    # CITE: https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
    # HELP: Learned how to suppress warnings
    with warnings.catch_warnings(action="ignore"):
        experiment_features(input_file = "abc.txt", output_file = 'results_1103.txt', first_k = 300)





if __name__ == "__main__":
    main()