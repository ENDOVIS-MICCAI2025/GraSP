# Experiment setup
FOLD="fold1" 
EXP_NAME="MVITv1_decoder"
TASK="TOOLS"
ARCH="MVIT"
CHECKPOINT="/data/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

#-------------------------
DATA_VER="psi-ava"
EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features.pth" 
FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE False \
TRAIN.ENABLE True \
DATA.NUM_FRAMES 16 \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
BN.NUM_BATCHES_PRECISE 72 \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.VAL_FEATURES_PATH $FF_VAL \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 28 \
MODEL.DECODER True \
OUTPUT_DIR $OUTPUT_DIR 

#################################################################################################################### 

# Experiment setup
FOLD="fold2" 
EXP_NAME="MVITv1_decoder"
TASK="TOOLS"
ARCH="MVIT"
CHECKPOINT="/data/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

#-------------------------
DATA_VER="psi-ava"
EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features.pth" 
FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE False \
TRAIN.ENABLE True \
DATA.NUM_FRAMES 16 \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
BN.NUM_BATCHES_PRECISE 72 \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.VAL_FEATURES_PATH $FF_VAL \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 28 \
MODEL.DECODER True \
OUTPUT_DIR $OUTPUT_DIR 

#####################################################################################################

# Experiment setup
FOLD="fold1" 
EXP_NAME="MVITv1_baseline_wd1e-6"
TASK="TOOLS"
ARCH="MVIT"
CHECKPOINT="/data/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

#-------------------------
DATA_VER="psi-ava"
EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features.pth" 
FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE False \
TRAIN.ENABLE True \
DATA.NUM_FRAMES 16 \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
BN.NUM_BATCHES_PRECISE 72 \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.VAL_FEATURES_PATH $FF_VAL \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 28 \
SOLVER.WEIGHT_DECAY 1e-6 \
OUTPUT_DIR $OUTPUT_DIR 