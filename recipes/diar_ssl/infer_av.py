# recipes/diar_ssl/infer_av.py

import os
import argparse
import toml
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict
from huggingface_hub import snapshot_download, hf_hub_download

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
        # Filter out harmless missing keys often found in WavLM loading
        real_missing = [k for k in missing if "masked_spec_embed" not in k]
        if real_missing:
            print(f"Warning: Missing keys: {real_missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected[:5]}...")
        
    model.eval()
    return model

def resolve_hub_path(path_or_id):
    path = Path(path_or_id)
    if path.exists():
        return path
    
    print(f"Path '{path_or_id}' not found locally. Attempting download from Hugging Face...")
    try:
        return Path(snapshot_download(repo_id=path_or_id))
    except Exception as e:
        raise ValueError(f"Could not find local path or download HF repo '{path_or_id}'. Error: {e}")

def resolve_embedding_model(path_or_id):
    if os.path.exists(path_or_id):
        return path_or_id
    
    if "wespeaker-voxceleb-resnet34-LM" in path_or_id:
        print("Downloading Embedding Model (wespeaker-voxceleb-resnet34-LM)...")
        return hf_hub_download(
            repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
            filename="pytorch_model.bin"
        )
    
    raise FileNotFoundError(f"Embedding model not found at {path_or_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AV Diarization Inference")
    parser.add_argument("-C", "--configuration", type=str, required=True, help="Path to model config.toml")
    parser.add_argument("-i", "--in_wav_scp", type=str, required=True, help="Path to wav.scp")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained checkpoint (folder or .bin)")
    parser.add_argument("--visual_root", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--diarizen_hub", type=str, required=True, help="Path to DiariZen hub (local path or HF Repo ID)")
    parser.add_argument("--embedding_model", type=str, required=True, help="Path to embedding model")
    
    # Inference params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--segmentation_step", type=float, default=0.1)
    parser.add_argument("--min_speakers", type=int, default=1)
    parser.add_argument("--max_speakers", type=int, default=20)
    
    # VBx Clustering params (Added to fix KeyError)
    parser.add_argument("--ahc_criterion", type=str, default="distance")
    parser.add_argument("--ahc_threshold", type=float, default=0.6)
    parser.add_argument("--Fa", type=float, default=0.07)
    parser.add_argument("--Fb", type=float, default=0.8)
    parser.add_argument("--lda_dim", type=int, default=128)
    parser.add_argument("--max_iters", type=int, default=20)
    
    args = parser.parse_args()

    # 1. Resolve paths
    hub_path = resolve_hub_path(args.diarizen_hub)
    embed_model_path = resolve_embedding_model(args.embedding_model)

    # 2. Load the trained model
    trained_model = load_trained_model(args.configuration, args.ckpt_path)

    # 3. Setup Pipeline Config
    # We must provide all VBx parameters because overriding clustering config wipes the defaults
    clustering_config = {
        "method": "VBxClustering",
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers,
        "ahc_criterion": args.ahc_criterion,
        "ahc_threshold": args.ahc_threshold,
        "Fa": args.Fa,
        "Fb": args.Fb,
        "lda_dim": args.lda_dim,
        "max_iters": args.max_iters
    }

    config_parse = {
        "inference": {
            "args": {
                "batch_size": args.batch_size, 
                "segmentation_step": args.segmentation_step, 
                "seg_duration": trained_model.chunk_size, 
                "apply_median_filtering": True
            }
        },
        "clustering": {"args": clustering_config} 
    }

    pipeline = AV_DiariZenPipeline(
        diarizen_hub=hub_path,
        embedding_model=embed_model_path,
        config_parse=config_parse,
        rttm_out_dir=args.out_dir,
        visual_root=args.visual_root
    )

    # 4. Inject the trained model into the pipeline
    pipeline._segmentation.model = trained_model
    pipeline._segmentation.model.to(pipeline._segmentation.device)

    # 5. Run Inference
    audio_dict = load_scp(args.in_wav_scp)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for sess, wav_path in audio_dict.items():
        print(f"Processing Session: {sess}")
        try:
            pipeline(wav_path, sess_name=sess)
        except Exception as e:
            # Print full stack trace for debugging
            import traceback
            traceback.print_exc()
            print(f"Error processing session {sess}: {e}")
        
    print(f"Inference complete. Results in {args.out_dir}")