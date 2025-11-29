# recipes/diar_ssl/infer_av.py
# Licensed under the MIT license.
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
import os
import cv2
import toml
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Dict, Any, Union

from scipy.ndimage import median_filter
from torch.utils.data import Dataset, DataLoader

from pyannote.audio.core.io import AudioFile
from pyannote.audio.utils.signal import Binarize, SlidingWindowFeature
from pyannote.core import SlidingWindow, Segment

from diarizen.pipelines.inference import DiariZenPipeline
from diarizen.pipelines.utils import scp2path

# Constants matching training
VISUAL_RESIZE_DIMS = (224, 224)
VISUAL_FPS = 25

class SlidingWindowAVDataset(Dataset):
    """
    Dataset to yield synchronized Audio-Visual chunks for inference.
    """
    def __init__(self, waveform: torch.Tensor, visual_path: str, chunk_size: float, step_size: float, sample_rate: int):
        self.waveform = waveform
        self.visual_path = visual_path
        self.chunk_size = chunk_size
        self.step_size = step_size
        self.sample_rate = sample_rate
        self.duration = waveform.shape[1] / sample_rate
        
        # Calculate number of chunks
        self.chunks = []
        start = 0.0
        while start + chunk_size <= self.duration:
            self.chunks.append(start)
            start += step_size
            
        # Add last chunk if needed (padding logic can be complex, skipping for simplicity or handling partials)
        if start < self.duration:
            self.chunks.append(start)

    def __len__(self):
        return len(self.chunks)

    def load_visual_frames(self, start_sec, end_sec):
        """Load and process video frames for a specific time window."""
        start_frame = int(start_sec * VISUAL_FPS)
        end_frame = int(end_sec * VISUAL_FPS)
        target_frames = end_frame - start_frame
        
        # Pre-allocate buffer (T, H, W, C)
        buffer = np.zeros((target_frames, *VISUAL_RESIZE_DIMS, 3), dtype=np.uint8)
        
        if os.path.exists(self.visual_path):
            cap = cv2.VideoCapture(self.visual_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for i in range(target_frames):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, VISUAL_RESIZE_DIMS)
                    buffer[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    break
            cap.release()
        
        # Convert to Tensor (T, C, H, W) and normalize
        visual_tensor = torch.from_numpy(buffer).float()
        visual_tensor = visual_tensor.permute(0, 3, 1, 2) / 255.0
        return visual_tensor

    def __getitem__(self, idx):
        start_time = self.chunks[idx]
        end_time = min(start_time + self.chunk_size, self.duration)
        
        # Audio slicing
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(start_time * self.sample_rate + self.chunk_size * self.sample_rate)
        
        wav_chunk = self.waveform[:, start_sample:end_sample]
        
        # Pad audio if necessary (at the end of file)
        target_samples = int(self.chunk_size * self.sample_rate)
        if wav_chunk.shape[1] < target_samples:
            wav_chunk = torch.nn.functional.pad(wav_chunk, (0, target_samples - wav_chunk.shape[1]))

        # Visual slicing
        vis_chunk = self.load_visual_frames(start_time, end_time)
        
        # Pad visual if necessary
        target_vis_frames = int(self.chunk_size * VISUAL_FPS)
        if vis_chunk.shape[0] < target_vis_frames:
             # Pad temporal dimension (dim 0)
             padding = torch.zeros((target_vis_frames - vis_chunk.shape[0], *vis_chunk.shape[1:]))
             vis_chunk = torch.cat([vis_chunk, padding], dim=0)

        return {
            "waveforms": wav_chunk,
            "visual": vis_chunk,
            "start_time": start_time
        }

def av_collate_inference(batch):
    return {
        "waveforms": torch.stack([b["waveforms"] for b in batch]),
        "visual": torch.stack([b["visual"] for b in batch]),
        "start_time": [b["start_time"] for b in batch]
    }

class AVDiariZenPipeline(DiariZenPipeline):
    def __init__(
        self, 
        visual_root: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_root = Path(visual_root)
        self.visual_fps = VISUAL_FPS

    def get_av_segmentations(self, in_wav, sess_name):
        """
        Manually run inference using the AVModel with synchronized audio/video inputs.
        Mimics pyannote.audio.core.inference.Inference aggregation logic.
        """
        waveform, sample_rate = torchaudio.load(in_wav)
        
        # Determine duration and paths
        video_path = self.visual_root / f"{sess_name}.mp4"
        if not video_path.exists():
            print(f"Warning: Video file not found for {sess_name} at {video_path}. Using black frames.")

        model = self._segmentation.model
        model.eval()
        device = self._segmentation.device
        model.to(device)

        # Create Dataset and DataLoader
        dataset = SlidingWindowAVDataset(
            waveform, 
            str(video_path), 
            chunk_size=self._segmentation.duration, 
            step_size=self._segmentation.step, 
            sample_rate=sample_rate
        )
        
        loader = DataLoader(
            dataset, 
            batch_size=self.segmentation_batch_size, 
            collate_fn=av_collate_inference,
            num_workers=4,
            pin_memory=True
        )

        # Prepare Aggregation Buffers
        # Get frame resolution from model introspection
        num_samples = int(self._segmentation.duration * sample_rate)
        num_frames = model.num_frames(num_samples)
        # Time per frame (resolution)
        resolution = model.specifications.duration / num_frames 
        
        total_duration = waveform.shape[1] / sample_rate
        total_frames = int(total_duration / resolution) + 100 # buffer
        
        num_classes = len(model.specifications.classes)
        if model.specifications.powerset:
            num_classes = model.specifications.num_powerset_classes

        # Aggregation buffers
        aggregated_scores = np.zeros((total_frames, num_classes))
        overlap_count = np.zeros((total_frames, 1))

        with torch.no_grad():
            for batch in loader:
                waveforms = batch["waveforms"].to(device)
                visual = batch["visual"].to(device)
                start_times = batch["start_time"]

                # Add channel dim to audio if needed by model (Batch, Channel, Time) -> (Batch, Time) handled inside?
                # AVModel expects (Batch, Channel, Time) for audio input
                # Dataset returns (Batch, Time). We need to unsqueeze channel.
                if waveforms.dim() == 2:
                    waveforms = waveforms.unsqueeze(1)

                inputs = {"waveforms": waveforms, "visual": visual}
                
                # Forward pass
                # Model output: (Batch, Frames, Classes)
                predictions = model(inputs)
                
                # Aggregate
                for i, pred in enumerate(predictions):
                    start_t = start_times[i]
                    # Determine start frame index in global buffer
                    start_idx = int(start_t / resolution)
                    
                    # Number of frames in this prediction
                    n_pred_frames = pred.shape[0]
                    
                    # Safe index bounds
                    end_idx = min(start_idx + n_pred_frames, total_frames)
                    trunc_pred = pred[:end_idx - start_idx].cpu().numpy()
                    
                    aggregated_scores[start_idx:end_idx] += trunc_pred
                    overlap_count[start_idx:end_idx] += 1

        # Average scores
        overlap_count[overlap_count == 0] = 1 # avoid div by zero
        final_scores = aggregated_scores / overlap_count
        
        # Crop to actual duration
        final_num_frames = int(total_duration / resolution)
        final_scores = final_scores[:final_num_frames]

        # Construct SlidingWindowFeature
        sw = SlidingWindow(start=0.0, step=resolution, duration=resolution)
        return SlidingWindowFeature(final_scores, sw)

    def __call__(self, in_wav, sess_name=None):
        assert isinstance(in_wav, (str, Path)), "Input must be a path for AV pipeline."
        
        print('Extracting AV segmentations.')
        # --- Custom AV Logic ---
        segmentations = self.get_av_segmentations(in_wav, sess_name)
        # -----------------------

        if self.apply_median_filtering:
            segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode='reflect')

        # binarize segmentation
        binarized_segmentations = segmentations     # powerset

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            binarized_segmentations,
            self._segmentation.model._receptive_field,
            warm_up=(0.0, 0.0),
        )

        print("Extracting Embeddings (Audio-Only).")
        # Load audio for embedding extraction (handled by parent class mostly, but we need waveform)
        waveform, sample_rate = torchaudio.load(in_wav)
        
        # Standard embedding extraction (Wespeaker is usually audio-only)
        embeddings = self.get_embeddings(
            {"waveform": waveform, "sample_rate": sample_rate},
            binarized_segmentations,
            exclude_overlap=self.embedding_exclude_overlap,
        )

        print("Clustering.")
        hard_clusters, _, _ = self.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            min_clusters=self.min_speakers,  
            max_clusters=self.max_speakers
        )

        count.data = np.minimum(count.data, self.max_speakers).astype(np.int8)

        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        hard_clusters[inactive_speakers] = -2
        
        discrete_diarization, _ = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )

        to_annotation = Binarize(
            onset=0.5,
            offset=0.5,
            min_duration_on=0.0,
            min_duration_off=0.0
        )
        result = to_annotation(discrete_diarization)
        result.uri = sess_name
        
        if self.rttm_out_dir is not None:
            assert sess_name is not None
            rttm_out = os.path.join(self.rttm_out_dir, sess_name + ".rttm")
            with open(rttm_out, "w") as f:
                f.write(result.to_rttm())
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "This script performs AV diarization using DiariZen AV pipeline",
        add_help=True,
        usage="%(prog)s [options]",
    )

    # Required paths
    parser.add_argument(
        "--in_wav_scp",
        type=str,
        required=True,
        help="Path to wav.scp."
    )
    parser.add_argument(
        "--diarizen_hub",
        type=str,
        required=True,
        help="Path to DiariZen model hub directory (containing config.toml and pytorch_model.bin)."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Path to pretrained embedding model (e.g. wespeaker)."
    )
    parser.add_argument(
        "--visual_root",
        type=str,
        required=True,
        help="Path to folder containing video files ({sess_name}.mp4)."
    )

    # inference parameters
    parser.add_argument(
        "--seg_duration",
        type=int,
        default=5, # Changed default to 5 to match typical AVModel training
        help="Segment duration in seconds (Chunk size).",
    )
    parser.add_argument(
        "--segmentation_step",
        type=float,
        default=0.1,
        help="Shifting ratio during segmentation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Input batch size for inference.",
    )
    parser.add_argument(
        "--apply_median_filtering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply median filtering to segmentation output.",
    )

    # clustering parameters
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="VBxClustering",
        choices=["VBxClustering", "AgglomerativeClustering"],
        help="Clustering method to use.",
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=1,
        help="Minimum number of speakers.",
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=20,
        help="Maximum number of speakers.",
    )
    parser.add_argument(
        "--ahc_criterion",
        type=str,
        default="distance",
        help="AHC criterion (for VBx).",
    )
    parser.add_argument(
        "--ahc_threshold",
        type=float,
        default=0.6,
        help="AHC threshold.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=13,
        help="Minimum cluster size (for AHC).",
    )
    parser.add_argument(
        "--Fa",
        type=float,
        default=0.07,
        help="VBx Fa parameter.",
    )
    parser.add_argument(
        "--Fb",
        type=float,
        default=0.8,
        help="VBx Fb parameter.",
    )
    parser.add_argument(
        "--lda_dim",
        type=int,
        default=128,
        help="VBx LDA dimension.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=20,
        help="VBx maximum iterations.",
    )
    parser.add_argument(
        "--rttm_out_dir",
        type=str,
        default=None,
        required=False,
        help="Path to output folder.",
    )

    args = parser.parse_args()
    print(args)

    # Prepare Config Overrides
    inference_config = {
        "seg_duration": args.seg_duration,
        "segmentation_step": args.segmentation_step,
        "batch_size": args.batch_size,
        "apply_median_filtering": args.apply_median_filtering
    }

    clustering_config = {
        "method": args.clustering_method,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers
    }
    if args.clustering_method == "AgglomerativeClustering":
        clustering_config.update({
            "ahc_threshold": args.ahc_threshold,
            "min_cluster_size": args.min_cluster_size
        })
    elif args.clustering_method == "VBxClustering":
        clustering_config.update({
            "ahc_criterion": args.ahc_criterion,
            "ahc_threshold": args.ahc_threshold,
            "Fa": args.Fa,
            "Fb": args.Fb,
            "lda_dim": args.lda_dim,
            "max_iters": args.max_iters
        })

    config_parse = {
        "inference": {"args": inference_config},
        "clustering": {"args": clustering_config}
    }

    # Initialize AV Pipeline
    diarizen_pipeline = AVDiariZenPipeline(
        visual_root=args.visual_root,
        diarizen_hub=Path(args.diarizen_hub),
        embedding_model=args.embedding_model,
        config_parse=config_parse,
        rttm_out_dir=args.rttm_out_dir
    )

    # Run
    audio_f = scp2path(args.in_wav_scp)
    for audio_file in audio_f:
        sess_name = Path(audio_file).stem.split('.')[0]
        print(f'Processing: {sess_name}')
        diarizen_pipeline(audio_file, sess_name=sess_name)