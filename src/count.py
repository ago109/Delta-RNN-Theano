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

infname = ''
options, remainder = getopt.getopt(sys.argv[1:], '', ["infname="])

# Collect arguments ifrom argv
for opt, arg in options:
    if opt in ("--infname"):
        infname = arg

numSymb = 0
with (open(infname, 'r')) as f:
    for line in f:
        numSymb += len(re.split(r"\s+",line))
print(" Num Symbols = ",numSymb)
