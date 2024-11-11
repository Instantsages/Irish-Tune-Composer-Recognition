"""
    File:        get_data.py
    Author:      Yifan Wu
    Project:     CS 330 - Irish Tune Composer Recognition
    Semester:    Fall 2024
    Description: Visualize data with image and text
"""


import pandas as pd
import re

def convert_to_txt(dest, source = "Tunes_Found.csv"):
    """
    Converts formatted csv data to formatted txt
    Keyword arguments:
        dest (string): Name of the txt file to store txt output
        source (string): Name of the csv file to read data from

    """
    tunes_data = pd.read_csv(source)
    tunes_data = tunes_data.reset_index()

    with open(dest, "w") as text_file:
        # Iterating through each tune
        for index, row in tunes_data.iterrows():
            musician = "composer: " + row["Composer/Contributor"] + "\n"
            abc_notation = row["ABC Notation"]

            # Replace newline indicator with real newline
            abc_notation = abc_notation.replace('\\n', '\n')
            # Remove places where multiple newlines repeat
            abc_notation = re.sub(r'\n+', '\n', abc_notation)

            # For all but the last tune, make sure that the tune ends with newline
            if not abc_notation.endswith('\n') and index < len(tunes_data) - 1:
                abc_notation = abc_notation + '\n'

            # Writes to txt file
            text_file.write(musician)
            text_file.write(abc_notation)

            # Remove newline after the last tune
            if index < len(tunes_data) - 1:
                text_file.write("\n")
    

# Controls operation of the program
def main():
    convert_to_txt("abc.txt")


if __name__ == "__main__":
    main()