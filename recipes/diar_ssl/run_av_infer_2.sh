export PYTHONPATH=$PYTHONPATH:.

python recipes/diar_ssl/infer_av_2.py \
    --in_wav_scp /home/ubuntu/Document/MMML/MSDWILD/data/val/wav.scp \
    --visual_root /home/ubuntu/Document/MMML/MSDWILD/mp4 \
    --diarizen_hub /home/ubuntu/Document/MMML/DiariZen/recipes/diar_ssl/exp/av_model_large/av_wavlm/checkpoints/best \
    --embedding_model pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin \
    --rttm_out_dir /home/ubuntu/Document/MMML/MSDWILD/inference_results/av_wavlm \
    --seg_duration 5 \
    --batch_size 8 \
    --clustering_method VBxClustering \
    --min_speakers 1 \
    --max_speakers 5