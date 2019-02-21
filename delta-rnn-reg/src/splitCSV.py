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

# Execute main program (since it will most likely be from the command line)
infname = '' # data source to build dictionary from
textfname = '' # name of output lexicion
idfname = ''
idmap =  ''
startTok="<start>"
endTok="<end>"

options, remainder = getopt.getopt(sys.argv[1:], '', ["infname=","textfname=","idfname=","idmap=","startTok=","endTok"])

# Collect arguments ifrom argv
for opt, arg in options:
    if opt in ("--infname"):
        infname = arg
    elif opt in ("--textfname"):
        textfname = arg
    elif opt in ("--idfname"):
        idfname = arg
    elif opt in ("--idmap"):
        idmap = arg
    elif opt in ("--startTok"):
        startTok = arg
    elif opt in ("--endTok"):
        endTok = arg

US="<us>"
tw = open(textfname, 'wb')
iw = open(idfname, 'wb')
idmap = open(idmap, 'wb')
imap = {} # empty dictionary
ptr = 0

numIDwords = 0
numWords = 0
numLines = 0
with open(infname,'r') as f:
    reader = csv.reader(f, delimiter=',',quotechar='"')
    for row in reader:
        if numLines > 0:
            ident = row[1].replace('"',"").strip() # user-id
            if ident in imap:
                ident = "{0}".format(imap[ident])
            else:
                imap[ident] = ptr
                idmap.write("{0},{1}".format(ident,ptr))
                idmap.write("\n")
                ident = "{0}".format(ptr)
                ptr += 1
            sent = row[2].replace('"',"").strip() # sent-text
            sent = sent.replace("_",US)
            #sent = re.sub(r"\s+", '_', sent)
            tok = re.split(r"\s+",sent)
            line = ""
            for word in tok:
                if word != US:
                    word_new = " ".join(word)
                else:
                    word_new = word
                if len(line) > 0:
                    line = "{0} _ {1}".format(line,word_new) # insert artificial spaces between characters (delim)
                else:
                    line = word_new
            line = "{0} {1} {2}".format(startTok,line,endTok)
            numChar = len(re.split(r"\s+",line)) # count num unique symbols
            numWords += numChar
            tw.write(line) # write sentence to text stream
            tw.write("\n")
            ident = "{0} ".format(ident)
            id_line = ident * numChar
            id_line = id_line[:-1] # nix last excess space character
            check = len(re.split(r"\s+",id_line))
            numIDwords += check
            iw.write(id_line)
            iw.write("\n")
            #print(id_line)
            #sys.exit(0)
        numLines += 1
        #print("\r   {0} lines processed..".format(numLines), end='')
    print("")
tw.close()
iw.close()
print(numWords)
print(numIDwords)
