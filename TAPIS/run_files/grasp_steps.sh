# Experiment setup
TRAIN_FOLD="fold1"
TEST_FOLD="fold2" 
EXP_PREFIX="metric"
TASK="STEPS"
ARCH="TAPIS"

#-------------------------
DATASET="GraSP"
EXPERIMENT_NAME=$EXP_PREFIX"/"$TRAIN_FOLD
CONFIG_PATH="configs/"$DATASET"/"$ARCH"_"$TASK".yaml"
FRAME_DIR="./data/"$DATASET"/frames"
OUTPUT_DIR="outputs/"$DATASET"/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="./data/"$DATASET"/frame_lists"
ANNOT_DIR="./data/"$DATASET"/annotations/"$TRAIN_FOLD
COCO_ANN_PATH="./data/"$DATASET"/annotations/"$TRAIN_FOLD"/"$TEST_FOLD"_long-term_anns.json"
CHECKPOINT="./data/GraSP/pretrained_models/"$TASK"/steps_"$TRAIN_FOLD".pyth"
TYPE="pytorch"

export PYTHONPATH=/home/nayobi/Endovis/GraSP/TAPIS/tapis:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=1 python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE True \
TRAIN.ENABLE False \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.TRAIN_LISTS $TRAIN_FOLD".csv" \
ENDOVIS_DATASET.TEST_LISTS $TEST_FOLD".csv" \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.TEST_COCO_ANNS $COCO_ANN_PATH \
ENDOVIS_DATASET.TRAIN_GT_BOX_JSON "train_long-term_anns.json" \
ENDOVIS_DATASET.TEST_GT_BOX_JSON $TEST_FOLD"_long-term_anns.json" \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.TEST_FEATURES_PATH $FF_TEST \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 24 \
OUTPUT_DIR $OUTPUT_DIR 