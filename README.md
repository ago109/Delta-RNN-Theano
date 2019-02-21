# Delta-RNN-Theano

Simple code to support the paper ["Learning Simpler Language Models with the Differential State Framework"](https://www.mitpressjournals.org/action/showCitFormats?doi=10.1162/neco_a_01017).
This code contains a set of linked Python scripts, built on top of Theano (Theano-0.8.0-py2.7, tested on Centos 7), to train and run various neural language models, particularly:
* Long Short Term Memory (LSTM)
* Gated Recurrent Unit (GRU)
* Elman Recurrent Network (RNN)
* The newly proposed Delta-RNN (from this paper)

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
        year = {2017}
    }


Note: This code has only been minimally cleaned and offers no guarantees for practical applications, we apologize in advance for any inconvenience this might cause. This code is primarily meant to used to train simple RNN-LMs (recurrent neural network language models) on basic text corpora as in the original paper (and the sample training file was used to test that the code successfully ran).
