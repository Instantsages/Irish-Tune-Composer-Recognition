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
    tunes_data = pd.read_csv(source)
    tunes_data = tunes_data.reset_index()

    with open(dest, "w") as text_file:
        for index, row in tunes_data.iterrows():
            musician = "composer: " + row["Composer/Contributor"] + "\n"
            abc_notation = row["ABC Notation"]

            abc_notation = abc_notation.replace('\\n', '\n')
            abc_notation = re.sub(r'\n+', '\n', abc_notation)

            if not abc_notation.endswith('\n') and index < len(tunes_data) - 1:
                abc_notation = abc_notation + '\n'

            text_file.write(musician)
            text_file.write(abc_notation)
            if index < len(tunes_data) - 1:
                text_file.write("\n")
    


def main():
    convert_to_txt("abc.txt")


if __name__ == "__main__":
    main()