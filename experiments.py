from cs_senior_seminar import * 

from itertools import chain, combinations

from sklearn.metrics import rand_score

import heapq

import warnings


# This adds to running time
# class case():
#     def __init__(self, feature_names, composers, labels):
#         assert len(composers) == len(labels)
        
#         self._feature_names = feature_names.copy()
#         self._composers = composers
#         self._labels = labels

#         self._rand_score = rand_score(composers, labels)

#     def get_rand_score(self):
#         return self._rand_score
        


# CITE: https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
# DESC: Source of this function for computing power set
def powerset_list(s):
    "powerset([1,2,3]) --> [() (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)]"
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))[1:]


def experiment_features(input_file = 'sample_abc.txt', output_file = 'results.txt', first_k = 30):
    abc_tunes = read_abcs(input_file)
    midi_tunes = convert_abc_to_midi(abc_tunes)

    global FEATURES

    power_features = powerset_list(FEATURES)

    assert len(power_features) > first_k

    result_heap = []

    last_popped = 0

    for i, current_set in enumerate(power_features):
        features = extract_features(midi_tunes, list(current_set))

        dataset, composers = create_dataset(features)

        num_composers = 3
        labels = k_means_clustering(dataset, num_composers)

        rand_index = rand_score(composers, labels)

        if rand_index > last_popped:
            print(f"Tested subset {i} with rand index {rand_index}; pushed to heap")
            heapq.heappush(result_heap, (rand_index, current_set))
        if len(result_heap) > first_k:
            last_popped, _ = heapq.heappop(result_heap)
        
    result_heap.sort()

    with open(output_file,"w") as file:
        for item in result_heap:
            file.write(f"{item}\n")
        

        

def main():
    # CITE: https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
    # HELP: Learned how to suppress warnings
    with warnings.catch_warnings(action="ignore"):
        experiment_features()





if __name__ == "__main__":
    main()