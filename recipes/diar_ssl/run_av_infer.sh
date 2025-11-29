# Set paths variables
EXP_DIR=/home/ubuntu/Document/MMML/DiariZen/recipes/diar_ssl/exp/av_model_large/av_wavlm
CKPT_PATH=$EXP_DIR/checkpoints/best  # Point to the checkpoint you want to use
# CKPT_PATH=$EXP_DIR/checkpoints/epoch_0002_2_3
CONFIG_PATH=/home/ubuntu/Document/MMML/DiariZen/recipes/diar_ssl/conf/av_wavlm.toml
VISUAL_ROOT=/home/ubuntu/Document/MMML/MSDWILD/mp4
WAV_SCP=/home/ubuntu/Document/MMML/MSDWILD/data/val/wav.scp
OUTPUT_DIR=$EXP_DIR/inference_results

# DiariZen Hub path (downloaded previously via snapshot_download or local)
# If you don't have this separate, you can point to the repo cache or a local copy
DIARIZEN_HUB=BUT-FIT/diarizen-wavlm-large-s80-md
EMBEDDING_MODEL=pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin

python /home/ubuntu/Document/MMML/DiariZen/recipes/diar_ssl/infer_av.py \
    -C $CONFIG_PATH \
    -i $WAV_SCP \
    -o $OUTPUT_DIR \
    --ckpt_path $CKPT_PATH \
    --visual_root $VISUAL_ROOT \
    --diarizen_hub $DIARIZEN_HUB \
    --embedding_model $EMBEDDING_MODEL


# python recipes/diar_ssl/infer_av.py \
#     -C $CONFIG_PATH \
#     -i $WAV_SCP \
#     -o $OUTPUT_DIR \
#     --ckpt_path $CKPT_PATH \
#     --visual_root $VISUAL_ROOT \
#     --diarizen_hub $DIARIZEN_HUB \
#     --embedding_model $EMBEDDING_MODEL

# Define your local paths
# CKPT_FOLDER=recipes/diar_ssl/exp/av_model_large/checkpoints/epoch_0010