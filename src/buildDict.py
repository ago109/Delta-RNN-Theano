# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, getopt, optparse
import re
import numpy as np
import cPickle as pkl
import csv
#import unicodecsv as csv # run pip2 install unicodecsv
import io
import UnicodeTools as uc

def utf_8_encoder(unicode_csv_data): # allows csv- to handle utf-8
    for line in unicode_csv_data:
        yield line.encode('utf-8')

# A very simple dictionary/lexicon constructor

def build(fname, delim='\\s+',unktoken=''): # default: split on space
    lexicon = {} # empty dictionary
    idx = 0
    numSentSeen = 0
    numSymbSeen = 0
    with (open(fname, 'r')) as f:
        for line in f:
            #symbols = line.split(delim)
            symbols = re.split(delim,line) # want to split on regex
            numSymbSeen += len(symbols)
            for w in range(0,len(symbols)):
                symbol = symbols[w]
                if (symbol in lexicon) == False and len(symbol) > 0:
                    lexicon[symbol] = idx
                    #print("{0} to {1} ".format(symbol,idx),end='')
                    idx += 1 # increment pointer for symbol
            numSentSeen += 1
            print("\r  {0} lines {1} symbols seen...".format(numSentSeen,numSymbSeen),end='')
    if len(unktoken) > 0:
        lexicon[unktoken] = idx
    return lexicon

# Execute main program (since it will most likely be from the command line)
inputfile = '' # data source to build dictionary from
dictfname = '' # name of output lexicion
delim = '\\s+' # default delim is space character
unktoken = ''

options, remainder = getopt.getopt(sys.argv[1:], '', ["inputfile=","dictfname=","delim=","unktoken="])

# Collect arguments ifrom argv
for opt, arg in options:
    if opt in ("--inputfile"):
        inputfile = arg
    elif opt in ("--dictfname"):
        dictfname = arg
    elif opt in ("--delim"):
        delim = arg
    elif opt in ("--unktoken"):
        unktoken = arg

print(" > Build dictionary from data-file: ",inputfile)
lexicon = build(inputfile, delim=delim,unktoken=unktoken)
#print(lexicon)
print("\n Dictionary size = ",len(lexicon))
print(" > Saving lexicon to: ", dictfname)
outName = dictfname + ".pkl" # we will pickle (serialize) this dictionary to disk
pkl.dump( lexicon, open( outName, "wb" ) )

# As a small sanity-check, also write a word-list to disk
#writer = csv.writer(io.open(dictfname + "-list.csv", 'wb'))
writer = open(dictfname + "-list.csv", 'wb')
#writer = uc.UnicodeWriter(open(dictfname + "-list.csv", 'wb'))
for key, value in lexicon.items():
    #writer.writerow([key]) #writer.writerow([key, value])
    writer.write(key)
    writer.write("\n")
writer.close()
