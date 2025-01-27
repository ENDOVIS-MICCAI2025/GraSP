# Experiment setup
TRAIN_FOLD="train"
TEST_FOLD="test"
EXP_PREFIX="ALL"
TASK="PHASES"
ARCH="TAPIS"
SAMPLING_RATE=4
ONLINE=True

#-------------------------
DATASET="Led"
EXPERIMENT_NAME=$EXP_PREFIX"_"$ARCH"_v1_Phases_Sequence"
CONFIG_PATH="configs/"$DATASET"/"$ARCH"_"$TASK".yaml"
FRAME_DIR="./data/"$DATASET"/frames" # Take the videos to this?
OUTPUT_DIR="outputs/"$DATASET"/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="./data/"$DATASET"/frame_lists"
ANNOT_DIR="./data/"$DATASET"/annotations/"
COCO_ANN_PATH="./data/"$DATASET"/annotations/test.json"
CHECKPOINT="/home/naparicioc/K400_MVIT_B_16x4_CONV.pyth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/GraSP/TAPIS/tapis:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3 python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 4 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE False \
TRAIN.ENABLE True \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.TRAIN_LISTS $TRAIN_FOLD".csv" \
ENDOVIS_DATASET.TEST_LISTS $TEST_FOLD".csv" \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.TEST_COCO_ANNS $COCO_ANN_PATH \
ENDOVIS_DATASET.TRAIN_GT_BOX_JSON $TRAIN_FOLD".json" \
ENDOVIS_DATASET.TEST_GT_BOX_JSON $TEST_FOLD".json" \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.TEST_FEATURES_PATH $FF_TEST \
TRAIN.BATCH_SIZE 180 \
TRAIN.DATASET $DATASET \
TEST.DATASET $DATASET \
TEST.BATCH_SIZE 180 \
DATA.SAMPLING_RATE $SAMPLING_RATE \
DATA.ONLINE $ONLINE \
OUTPUT_DIR $OUTPUT_DIR 