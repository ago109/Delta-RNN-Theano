#!/bin/bash

INPUT_FILE="" # point to directory and file name of text file containing line-by-line corpus
DICT_FNAME="" # /path/to/output_dict_fname
DELIM="\\s+" # delimiter separating tokens in line-by-line input file
UNK_TOKEN="" # unknown token, e.g., <unk>, or leave blank/empty if no OOV token type is used

# run script to build lexicon
python src/buildDict.py --inputfile="$INPUT_FILE" --dictfname="$DICT_FNAME" --delim="$DELIM" --unktoken="$UNK_TOKEN" 
