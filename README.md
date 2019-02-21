# Delta-RNN-Theano

Simple code to support the paper ["Learning Simpler Language Models with the Differential State Framework"](Ororbia, Mikolov, and Reitter, 2017, Neural Computation) https://www.mitpressjournals.org/action/showCitFormats?doi=10.1162/neco_a_01017.
This code contains a set of linked Python scripts, built on top of Theano, to train and run various neural language models, particularly:
* Long Short Term Memory (LSTM)
* Gated Recurrent Unit (GRU)
* Elman Recurrent Network (RNN)
* The newly proposed Delta-RNN (in this paper)

An example file, for training a Delta-RNN with 100 hidden cells/units, is provided. To create the needed dictionary/lexicon, you can simply run the Bash script "build_dict.sh" filling in the relevant variable names. Make sure your text corpus (no matter what split it is, assuming you have a training text file, validation text file, and a test-set text file) is line-by-line (i.e., one sentence per line) and that tokens (words, subwords, or characters) are separated by a unique/consistent delimiter (default assumption is empty space, or "\\s+"). Make sure you add in any out-of-vocabulary (OOV) tokens that occur in your data (since this code-base assumes you have pre-processed your text to properly handle OOV cases).
There is some additional code in the /src folder that contains a procText.py script you can appropriate for pre-pending and appending <start> and <end> tokens if need be.
  
If you use this code or the Delta-RNN model itself (or even better, extend it!), please consider citing:

@article{ororbia2017deltarnn,
author = {Ororbia II, Alexander G. and Mikolov, Tomas and Reitter, David},
title = {Learning Simpler Language Models with the Differential State Framework},
journal = {Neural Computation},
volume = {29},
number = {12},
pages = {3327-3352},
year = {2017},
doi = {10.1162/neco\_a\_01017},
    note ={PMID: 28957029}
}


Note: This code has only been minimally cleaned and offers no guarantees for practical applications. This is primarily meant to used to train simple RNN-LMs (recurrent neural network language models) on basic text corpora as in the original paper.
