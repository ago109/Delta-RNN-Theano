#!/bin/bash
# In support of NECO submission Ororbia et al., 2017
# Script: fits a Delta-RNN model to some text data

# Set program meta-parameters
CNMEM_CAPACITY=0.97 #0.45
DEVICE=gpu0
SEED=1234
FIT_MODEL=1
TRAIN="data/tmn_word/tmn.word.train.txt"
VALID="data/tmn_word/tmn.word.valid.txt"
DICT="data/tmn_word/lex.pkl"
MODEL="drnn"
WGHT_SD=0.05
OUT_WGHT_SD=0.05
USE_FIFO_INIT=0
USE_ORTHO_INIT=0
N_H=100
N_EPOCHS=50
LR=0.002 #0.0001
LR_DECAY=0.5
FUNTYPE="tanh"
USE_LN=0
DROP_P=0 #0.2 #0.8 #0.2
DROP_INNER=0
BATCH_SIZE=64 # mini-batch size
UPDATER="adam"
REC_THETA_NAME=""
OUT_THETA_NAME=""
MODEL_PATH="/root/sharedfolder/ago109/models/tmn_word/drnn100/"
ERROR_MARK=100
GRAD_CLIP=-5 #-5
PATIENCE=1
NORM_MAX=-5.0

rm $MODEL_PATH* # clear out contents in old folder from last run
# Execute main program/simulation
THEANO_FLAGS=mode=FAST_RUN,device="$DEVICE",floatX=float32,lib.cnmem=$CNMEM_CAPACITY python delta-rnn-reg/src/train.py --trainfile="$TRAIN" --validfile="$VALID" --dictfname="$DICT" --model="$MODEL" --n_h="$N_H" --n_epochs="$N_EPOCHS" --lr="$LR" --batch_size="$BATCH_SIZE" --updater="$UPDATER" --recThetaName="$REC_THETA_NAME" --outThetaName="$OUT_THETA_NAME" --modelPath="$MODEL_PATH" --error_mark="$ERROR_MARK" --fit_model=$FIT_MODEL --wght_sd=$WGHT_SD --out_wght_sd=$OUT_WGHT_SD --useLN=$USE_LN --drop_p=$DROP_P --grad_clip=$GRAD_CLIP --patience=$PATIENCE --funtype=$FUNTYPE --norm_max=$NORM_MAX --useFanInFanOut=$USE_FIFO_INIT --lr_decay=$LR_DECAY --useOrtho=$USE_ORTHO_INIT --seed=$SEED --drop_inner=$DROP_INNER
