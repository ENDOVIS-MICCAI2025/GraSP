# ########################################################################################################################

# Experiment setup
FOLD="test" 
EXP_NAME="MVITv1_dense_mf_swinl"
TASK="ACTIONS"
ARCH="MVIT"
CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/fold2/checkpoint_best_phases.pyth"

#-------------------------
DATA_VER="psi-ava_dense"
EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features_mask2former_swinl.pth" 
FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features_mask2former_swinl.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python -B tools/run_net.py \
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
ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON "train_coco_preds_mask2former_swinl.json" \
ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON "val_coco_preds_mask2former_swinl.json" \
BN.NUM_BATCHES_PRECISE 72 \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.VAL_FEATURES_PATH $FF_VAL \
TEST.DATASET psi_ava_dense \
TRAIN.DATASET psi_ava_dense \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 28 \
MODEL.DECODER True \
OUTPUT_DIR $OUTPUT_DIR 

# # ########################################################################################################################

# # Experiment setup
# FOLD="fold2" 
# EXP_NAME="MVITv1_dense_mf_swinl"
# TASK="ACTIONS"
# ARCH="MVIT"
# CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

# #-------------------------
# DATA_VER="psi-ava_dense"
# EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
# CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
# FRAME_DIR="./data" 
# OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
# FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
# ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
# COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
# FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features_mask2former_swinl.pth" 
# FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features_mask2former_swinl.pth"

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

# mkdir -p $OUTPUT_DIR

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python -B tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# TRAIN.ENABLE True \
# DATA.NUM_FRAMES 16 \
# ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
# ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
# ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
# ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
# ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON "train_coco_preds_mask2former_swinl.json" \
# ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON "val_coco_preds_mask2former_swinl.json" \
# BN.NUM_BATCHES_PRECISE 72 \
# FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
# FEATURES.VAL_FEATURES_PATH $FF_VAL \
# TEST.DATASET psi_ava_dense \
# TRAIN.DATASET psi_ava_dense \
# TRAIN.BATCH_SIZE 24 \
# TEST.BATCH_SIZE 28 \
# MODEL.DECODER True \
# OUTPUT_DIR $OUTPUT_DIR 

# # ########################################################################################################################

# # Experiment setup
# FOLD="fold1" 
# EXP_NAME="MVITv1_dense_mf_r50"
# TASK="ACTIONS"
# ARCH="MVIT"
# CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

# #-------------------------
# DATA_VER="psi-ava_dense"
# EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
# CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
# FRAME_DIR="./data" 
# OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
# FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
# ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
# COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
# FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features_mask2former_r50.pth" 
# FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features_mask2former_r50.pth"

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

# mkdir -p $OUTPUT_DIR

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python -B tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# TRAIN.ENABLE True \
# DATA.NUM_FRAMES 16 \
# ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
# ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
# ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
# ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
# ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON "train_coco_preds_mask2former_r50.json" \
# ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON "val_coco_preds_mask2former_r50.json" \
# BN.NUM_BATCHES_PRECISE 72 \
# FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
# FEATURES.VAL_FEATURES_PATH $FF_VAL \
# TEST.DATASET psi_ava_dense \
# TRAIN.DATASET psi_ava_dense \
# TRAIN.BATCH_SIZE 24 \
# TEST.BATCH_SIZE 28 \
# MODEL.DECODER True \
# OUTPUT_DIR $OUTPUT_DIR 

# # ########################################################################################################################

# # Experiment setup
# FOLD="fold2" 
# EXP_NAME="MVITv1_dense_mf_r50"
# TASK="ACTIONS"
# ARCH="MVIT"
# CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

# #-------------------------
# DATA_VER="psi-ava_dense"
# EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
# CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
# FRAME_DIR="./data" 
# OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
# FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
# ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
# COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
# FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features_mask2former_r50.pth" 
# FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features_mask2former_r50.pth"

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

# mkdir -p $OUTPUT_DIR

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python -B tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# TRAIN.ENABLE True \
# DATA.NUM_FRAMES 16 \
# ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
# ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
# ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
# ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
# ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON "train_coco_preds_mask2former_r50.json" \
# ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON "val_coco_preds_mask2former_r50.json" \
# BN.NUM_BATCHES_PRECISE 72 \
# FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
# FEATURES.VAL_FEATURES_PATH $FF_VAL \
# TEST.DATASET psi_ava_dense \
# TRAIN.DATASET psi_ava_dense \
# TRAIN.BATCH_SIZE 24 \
# TEST.BATCH_SIZE 28 \
# MODEL.DECODER True \
# OUTPUT_DIR $OUTPUT_DIR 

# ##################################################################################################################
# # Experiment setup
# FOLD="fold1" 
# EXP_NAME="MVITv1_dense_mask"
# TASK="ACTIONS"
# ARCH="MVIT"
# CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

# #-------------------------
# DATA_VER="psi-ava_dense"
# EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
# CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
# FRAME_DIR="./data" 
# OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
# FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
# ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
# COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
# FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features_mask.pth" 
# FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features_mask.pth"

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

# mkdir -p $OUTPUT_DIR

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python -B tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# TRAIN.ENABLE True \
# DATA.NUM_FRAMES 16 \
# ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
# ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
# ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
# ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
# ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON "train_coco_preds_mask.json" \
# ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON "val_coco_preds_mask.json" \
# TEST.DATASET psi_ava_dense \
# TRAIN.DATASET psi_ava_dense \
# BN.NUM_BATCHES_PRECISE 72 \
# FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
# FEATURES.VAL_FEATURES_PATH $FF_VAL \
# TRAIN.BATCH_SIZE 24 \
# TEST.BATCH_SIZE 28 \
# MODEL.DECODER True \
# FEATURES.MODEL faster \
# OUTPUT_DIR $OUTPUT_DIR 

# ################################################################################################################### 

# # Experiment setup
# FOLD="fold2" 
# EXP_NAME="MVITv1_dense_mask"
# TASK="ACTIONS"
# ARCH="MVIT"
# CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

# #-------------------------
# DATA_VER="psi-ava_dense"
# EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
# CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
# FRAME_DIR="./data" 
# OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
# FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
# ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
# COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
# FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features_mask.pth" 
# FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features_mask.pth"

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

# mkdir -p $OUTPUT_DIR

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python -B tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# TRAIN.ENABLE True \
# DATA.NUM_FRAMES 16 \
# ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
# ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
# ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
# ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
# ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON "train_coco_preds_mask.json" \
# ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON "val_coco_preds_mask.json" \
# TEST.DATASET psi_ava_dense \
# TRAIN.DATASET psi_ava_dense \
# BN.NUM_BATCHES_PRECISE 72 \
# FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
# FEATURES.VAL_FEATURES_PATH $FF_VAL \
# TRAIN.BATCH_SIZE 24 \
# TEST.BATCH_SIZE 28 \
# MODEL.DECODER True \
# FEATURES.MODEL faster \
# OUTPUT_DIR $OUTPUT_DIR 

# # ########################################################################################################################

# # Experiment setup
# FOLD="fold2" 
# EXP_NAME="MVITv1_dense"
# TASK="ACTIONS"
# ARCH="MVIT"
# CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

# #-------------------------
# DATA_VER="psi-ava_dense"
# EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
# CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
# FRAME_DIR="./data" 
# OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
# FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
# ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
# COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
# FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features.pth" 
# FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features.pth"

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

# mkdir -p $OUTPUT_DIR

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python -B tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# TRAIN.ENABLE True \
# DATA.NUM_FRAMES 16 \
# ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
# ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
# ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
# ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
# ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON "train_coco_preds.json" \
# ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON "val_coco_preds.json" \
# BN.NUM_BATCHES_PRECISE 72 \
# FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
# FEATURES.VAL_FEATURES_PATH $FF_VAL \
# TEST.DATASET psi_ava_dense \
# TRAIN.DATASET psi_ava_dense \
# TRAIN.BATCH_SIZE 24 \
# TEST.BATCH_SIZE 28 \
# MODEL.DECODER True \
# OUTPUT_DIR $OUTPUT_DIR 