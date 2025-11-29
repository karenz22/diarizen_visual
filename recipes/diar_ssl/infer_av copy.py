# recipes/diar_ssl/infer_av.py

import os
import argparse
import toml
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict

# Import your AV components
from diarizen.models.eend.model_av_wavlm_conformer import AVModel
from diarizen.pipelines.inference_av import AV_DiariZenPipeline
from diarizen.pipelines.utils import scp2path

def load_scp(scp_file: str) -> Dict[str, str]:
    lines = [line.strip().split(None, 1) for line in open(scp_file)]
    return {x[0]: x[1] for x in lines}

def load_trained_model(config_path, ckpt_path):
    """
    Instantiate AVModel from config and load weights from checkpoint.
    """
    config = toml.load(config_path)
    model_args = config["model"]["args"]
    
    # Instantiate the model architecture
    print(f"Instantiating AVModel with args: {model_args}")
    model = AVModel(**model_args)
    
    # Load weights
    print(f"Loading checkpoint from: {ckpt_path}")
    if os.path.isdir(ckpt_path):
        # Handle accelerate checkpoint folder
        ckpt_path = os.path.join(ckpt_path, "pytorch_model.bin")
        
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Handle state_dict unpacking if necessary
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Fix potential key prefixes from DDP/Accelerator
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # Load into model
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected[:5]}...")
        
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AV Diarization Inference")
    parser.add_argument("-C", "--configuration", type=str, required=True, help="Path to model config.toml")
    parser.add_argument("-i", "--in_wav_scp", type=str, required=True, help="Path to wav.scp")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained checkpoint (folder or .bin)")
    parser.add_argument("--visual_root", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--diarizen_hub", type=str, required=True, help="Path to DiariZen hub (for config/support files)")
    parser.add_argument("--embedding_model", type=str, required=True, help="Path to embedding model")
    
    # Inference params (optional overrides)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--segmentation_step", type=float, default=0.1)
    parser.add_argument("--min_speakers", type=int, default=1)
    parser.add_argument("--max_speakers", type=int, default=20)
    
    args = parser.parse_args()

    # 1. Load the trained model
    trained_model = load_trained_model(args.configuration, args.ckpt_path)

    # 2. Setup Pipeline
    # We pass a dummy 'config_parse' to override specific params if needed
    config_parse = {
        "inference": {"args": {"batch_size": args.batch_size, "segmentation_step": args.segmentation_step, "seg_duration": trained_model.chunk_size, "apply_median_filtering": True}},
        "clustering": {"args": {"method": "VBxClustering", "min_speakers": args.min_speakers, "max_speakers": args.max_speakers}} # Defaults, can be expanded
    }

    pipeline = AV_DiariZenPipeline(
        diarizen_hub=Path(args.diarizen_hub),
        embedding_model=args.embedding_model,
        config_parse=config_parse,
        rttm_out_dir=args.out_dir,
        visual_root=args.visual_root
    )

    # 3. Inject the trained model into the pipeline
    # This overrides the default model loaded by the pipeline
    pipeline._segmentation.model = trained_model
    pipeline._segmentation.model.to(pipeline._segmentation.device)

    # 4. Run Inference
    audio_dict = load_scp(args.in_wav_scp)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for sess, wav_path in audio_dict.items():
        print(f"Processing Session: {sess}")
        # The pipeline class handles visual loading internally based on sess name
        pipeline(wav_path, sess_name=sess)
        
    print(f"Inference complete. Results in {args.out_dir}")