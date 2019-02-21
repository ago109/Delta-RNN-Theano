#!/bin/bash
# In support of NECO article "Learning Simpler Language Models with the Differential State Framework", Ororbia et al., 2017
# Script: fits a Delta-RNN model to some text data in line-by-line corpus format
# @author - Alexander G. Ororbia II

# Set program high-level meta-parameters
CNMEM_CAPACITY=0.97 #0.45
DEVICE=gpu0
SEED=1234
FIT_MODEL=1 # set to 0, if pure model inference is needed (i.e., evaluation on a test set <-- in this case point VALID to the location of test set)
TRAIN="/path/to/train.set.txt" # NOTE: can use whatever fnames you want, just be sure to use *.txt extension
VALID="/path/to/valid.set.txt" # NOTE: could also point to "test.set.txt" if you set FIT_MODEL=0
DICT="/path/to/lex.pkl" # running src/buildDict.py should give you the relevant Pickle dictionary file
MODEL="drnn" # model type: drnn, lstm, gru, rnn

# Random Initialization of Parameters (non-bias params)
# control std dev of Gaussian initialization
WGHT_SD=0.05
OUT_WGHT_SD=0.05
# if set to 1, use fan-in-fan-out initialization
USE_FIFO_INIT=0
# if set to 1, use random orthogonal initialization
USE_ORTHO_INIT=0
# note that FIFO takes precedence over Ortho and Ortho over Gaussian, so if Gaussian desired, set FIFO & ORTHO to 0

# Model meta-parameters
N_H=100 # number of cells/units
FUNTYPE="tanh" # for the Delta-RNN, this changes the mixing/outer function (ones to try are identity, relu, and tanh)
USE_LN=0 # if set to 1, uses layer normalization (experimental - no guarantees)
DROP_P=0 # don't use this...
DROP_INNER=0 # inner drop-out rate (works well for delta-rnn), i.e., 0.2 or 0.5

# Training meta-parameters
N_EPOCHS=50 # number of epochs
LR=0.002 # learning rate / global step size
LR_DECAY=0.5 # learning rate decay
BATCH_SIZE=64 # mini-batch size
UPDATER="adam" # update/optimization rule --> options = adam, sgd
REC_THETA_NAME="" # /path/to/pretrained_rec_params.npz  (shaped correctly)
OUT_THETA_NAME="" # /path/to/pretrained_out_params.npz  (shaped correctly)
MODEL_PATH="/root/sharedfolder/ago109/models/tmn_word/drnn100/" # output directory where model parameters will be saved (as npz files)
ERROR_MARK=100 # marker / checkpoint for measuring validation loss/error
GRAD_CLIP=-5 # if > 0, will hard clip the gradients (to prevent exploding gradients)
PATIENCE=1 # don't use this...
NORM_MAX=-5.0 # if > 0, will employ gradient re-scaling (projecting gradients each to a Gaussian ball of radius NORM_MAX)


# Execute main program/simulation
rm $MODEL_PATH* # clear out contents in old folder from last run (comment this out if you are worried about old information being deleted, this is to clean out junk from useless prior runs)

THEANO_FLAGS=mode=FAST_RUN,device="$DEVICE",floatX=float32,lib.cnmem=$CNMEM_CAPACITY python delta-rnn-reg/src/train.py --trainfile="$TRAIN" --validfile="$VALID" --dictfname="$DICT" --model="$MODEL" --n_h="$N_H" --n_epochs="$N_EPOCHS" --lr="$LR" --batch_size="$BATCH_SIZE" --updater="$UPDATER" --recThetaName="$REC_THETA_NAME" --outThetaName="$OUT_THETA_NAME" --modelPath="$MODEL_PATH" --error_mark="$ERROR_MARK" --fit_model=$FIT_MODEL --wght_sd=$WGHT_SD --out_wght_sd=$OUT_WGHT_SD --useLN=$USE_LN --drop_p=$DROP_P --grad_clip=$GRAD_CLIP --patience=$PATIENCE --funtype=$FUNTYPE --norm_max=$NORM_MAX --useFanInFanOut=$USE_FIFO_INIT --lr_decay=$LR_DECAY --useOrtho=$USE_ORTHO_INIT --seed=$SEED --drop_inner=$DROP_INNER
