"""
    File:        visualization.py
    Authors:     Ahmed Abdelrehim, Maheen Masoud, and Yifan Wu
    Project:     CS 330 - Irish Tune Composer Recognition
    Semester:    Fall 2024
    Description: Visualize data with image and text
"""

from cs_senior_seminar import *

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import statistics


def scatterplot_2items(features, labels, x_item, y_item):
    """
    Make a scatter plot with two specified dimensions

    Keyword arguments:
        features (list of dict): Keeps track of the features of the tunes to look at
        labels (list): Labels that specify clusters of data points
    """
    tunes_list = []
    for composer in features:
        tunes_list.extend(features[composer])

    plot_data = pd.DataFrame(tunes_list)

    assert x_item in plot_data, f"{x_item} is not in the data given"
    assert y_item in plot_data, f"{y_item} is not in the data given"

    sns.scatterplot(data=plot_data, x=x_item, y=y_item, hue=labels)

    for composer in features:
        current_tunes = features[composer]

        for tune in current_tunes:
            plt.annotate(composer, (tune[x_item], tune[y_item]))

    print(f"Displaying features: {x_item}, {y_item}. \n")

    plt.show()


def analyze_features(features, labels):
    """
    Group data points by cluster, calculating the means and standard deviations of each feature

    Keyword arguments:
        features (list of dict): Keeps track of the features of the tunes to look at
        labels (list): Labels that specify clusters of data points

    Returns: a dictionary that is indexed by cluster label and then feature name
             For each feature within each cluster, the dictionary contains the mean and standard deviation
    """

    analysis = {}

    for composer in features:

        current_tunes = features[composer]

        # Organize data by cluster for analysis
        for index in range(len(current_tunes)):
            current_cluster = labels[index]
            current_feature = current_tunes[index]

            if not current_cluster in analysis:
                analysis[current_cluster] = {}

            for feature_name in current_feature:
                if not feature_name in analysis[current_cluster]:
                    analysis[current_cluster][feature_name] = []

                analysis[current_cluster][feature_name].append(
                    current_feature[feature_name]
                )

    # Keep track of the mean and sd within each group
    for current_cluster in analysis:
        current_feature = analysis[current_cluster]

        for feature_name in current_feature:
            analysis[current_cluster][feature_name] = (
                statistics.mean(current_feature[feature_name]),
                statistics.stdev(current_feature[feature_name]),
            )

    return analysis


def find_analysis(analysis, cluster_label, feature_name):
    """
    Prints out the average and standard deviation of a feature within a cluster

    Keyword arguments:
        analysis: dictionary that is output from analyze_features. Contains the mean and standard deviation
                  for each feature within each cluster
        cluster_label: the cluster to look at
        feature_name (string): the name of the feature to look at
    """
    mean, sd = analysis[cluster_label][feature_name]

    print(f"Looking at Cluster {cluster_label} and Feature {feature_name}")
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {sd}\n")


def demo():
    input_file = "abc.txt"

    abc_tunes = read_abcs(input_file)
    midi_tunes = convert_abc_to_midi(abc_tunes)

    features = extract_features(midi_tunes)
    dataset, composers = create_dataset(features)
    # print(dataset)

    labels = k_means_clustering(dataset, 8)

    scatterplot_2items(features, labels, "avg_duration", "contour_down")
    # scatterplot_2items(features, labels, 'different_rhythms', 'contour_up')

    feature_analysis = analyze_features(features, labels)

    find_analysis(feature_analysis, 0, "notes")

    input("Push enter to continue \n")

    for cluster_label in feature_analysis:
        for feature_name in feature_analysis[cluster_label]:
            find_analysis(feature_analysis, cluster_label, feature_name)


def main():
    demo()

    # for cluster_label in feature_analysis:
    #     for feature_name in feature_analysis[cluster_label]:
    #         find_analysis(feature_analysis, cluster_label, feature_name)


if __name__ == "__main__":
    main()
