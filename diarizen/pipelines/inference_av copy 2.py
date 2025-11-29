# from diarizen.pipelines.inference import DiariZenPipeline
# from diarizen.models.eend.model_av_wavlm_conformer import AVModel
# import torch
# import torchaudio
# import numpy as np
# from pyannote.audio.utils.signal import SlidingWindow

# class AV_DiariZenPipeline(DiariZenPipeline):
#     def __init__(self, visual_root, **kwargs):
#         super().__init__(**kwargs)
#         self.visual_root = visual_root
        
#         # Need to ensure self._segmentation.model is our AVModel
#         # This is usually set via the pretrained loading, but here we enforce it

#     def get_av_segmentations(self, audio_path, session_name):
#         """
#         Custom sliding window loop for Audio-Visual Segmentation
#         """
#         model = self._segmentation.model
#         device = self._segmentation.device
        
#         # Audio loading
#         waveform, sample_rate = torchaudio.load(audio_path)
#         duration = waveform.shape[1] / sample_rate
        
#         # Define Sliding Window
#         chunk_duration = model.specifications.duration
#         step = self._segmentation.segmentation_step * chunk_duration
#         sliding_window = SlidingWindow(start=0, duration=chunk_duration, step=step)
        
#         outputs = []
        
#         # Loop over chunks
#         for chunk in sliding_window(torch.zeros(1, waveform.shape[1]), sample_rate):
#             # 1. Extract Audio Chunk
#             start_sample = int(chunk.start * sample_rate)
#             end_sample = int(chunk.end * sample_rate)
            
#             if end_sample > waveform.shape[1]: break
            
#             wav_chunk = waveform[:, start_sample:end_sample].unsqueeze(0).to(device)
            
#             # 2. Extract Visual Chunk (Use the logic from Dataset)
#             # This requires access to the dataset loader or duplication of logic
#             visual_chunk = self.load_visual_chunk(session_name, chunk.start, chunk.end).unsqueeze(0).to(device)
            
#             # 3. Forward
#             with torch.no_grad():
#                 inputs = {"waveforms": wav_chunk, "visual": visual_chunk}
#                 # Shape: (1, num_frames, num_classes)
#                 prediction = model(inputs)
                
#             outputs.append((chunk, prediction[0].cpu().numpy()))

#         # Aggregation logic (Overlap-Add) would go here
#         # For simplicity, returning the raw list or a simplified aggregation
#         return self._aggregate(outputs, duration)

#     def load_visual_chunk(self, session, start, end):
#         # ... Implement OpenCV loading similar to Dataset ...
#         pass
        
#     def __call__(self, in_wav, sess_name=None):
#         # Override the segmentation extraction part
#         print('Extracting AV segmentations.')
        
#         # Use custom AV segmentation instead of self.get_segmentations
#         segmentations = self.get_av_segmentations(in_wav, sess_name)
        
#         if self.apply_median_filtering:
#              from scipy.ndimage import median_filter
#              segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode='reflect')

#         binarized_segmentations = segmentations

#         # The rest of the pipeline (clustering, etc.) can remain similar 
#         # provided `binarized_segmentations` is in the correct `SlidingWindowFeature` format.
        
#         # ... (rest of the logic from original inference.py) ...
#         # Note: Embedding extraction might also need AV support if you want visual embeddings for clustering.
#         # If keeping WavLM embeddings for clustering, original logic holds.
        
#         return super().__call__(in_wav, sess_name)

# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
import os
import cv2
import toml
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from scipy.ndimage import median_filter
from pyannote.audio.core.io import AudioFile
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.audio.utils.signal import Binarize
from pyannote.database.protocol.protocol import ProtocolFile
from huggingface_hub import snapshot_download, hf_hub_download

# Import base pipeline and utils
from diarizen.pipelines.inference import DiariZenPipeline
from diarizen.pipelines.utils import scp2path

class AV_DiariZenPipeline(DiariZenPipeline):
    """
    Audio-Visual Diarization Pipeline.
    Extends DiariZenPipeline to support visual input for segmentation.
    """
    def __init__(
        self, 
        visual_root: str,
        visual_fps: int = 25,
        visual_size: Tuple[int, int] = (224, 224),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_root = visual_root
        self.visual_fps = visual_fps
        self.visual_size = visual_size
        
        # Ensure the segmentation model is in eval mode and on the correct device
        self._segmentation.model.eval()

    @classmethod
    def from_pretrained(
        cls, 
        repo_id: str, 
        visual_root: str, # Added argument
        cache_dir: str = None,
        rttm_out_dir: str = None,
    ) -> "AV_DiariZenPipeline":
        
        # Download DiariZen hub files
        diarizen_hub = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )

        # Download Embedding model (assuming audio-only embedding for clustering)
        embedding_model = hf_hub_download(
            repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
            filename="pytorch_model.bin",
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )

        return cls(
            diarizen_hub=Path(diarizen_hub).expanduser().absolute(),
            embedding_model=embedding_model,
            rttm_out_dir=rttm_out_dir,
            visual_root=visual_root
        )

    def load_visual_chunk(self, session: str, start: float, end: float) -> torch.Tensor:
        """
        Load visual frames for a specific time window.
        """
        start_frame = int(start * self.visual_fps)
        end_frame = int(end * self.visual_fps)
        num_frames_target = end_frame - start_frame
        
        frames = []
        video_path = os.path.join(self.visual_root, f"{session}.mp4")
        
        # Simple cv2 frame loading
        # Note: For high throughput, consider decord or a persistent VideoReader object
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for _ in range(num_frames_target):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, self.visual_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    # Padding if video ends early
                    frames.append(np.zeros((*self.visual_size, 3), dtype=np.uint8))
            cap.release()
        else:
            # Handle missing video files gracefully (zero padding)
            # print(f"Warning: Video file not found for {session}, utilizing zero padding.")
            frames = np.zeros((num_frames_target, *self.visual_size, 3), dtype=np.uint8)

        if not frames:
             frames = np.zeros((num_frames_target, *self.visual_size, 3), dtype=np.uint8)

        # Convert to Tensor: (T, C, H, W)
        visual_tensor = torch.tensor(np.array(frames), dtype=torch.float32)
        visual_tensor = visual_tensor.permute(0, 3, 1, 2) / 255.0
        
        return visual_tensor

    def get_av_segmentations(self, waveform: torch.Tensor, sample_rate: int, sess_name: str) -> SlidingWindowFeature:
        """
        Custom sliding window inference loop for Audio-Visual Segmentation.
        Performs Overlap-Add aggregation.
        """
        model = self._segmentation.model
        device = self._segmentation.device
        
        # 1. Setup Sliding Window
        duration = waveform.shape[1] / sample_rate
        chunk_duration = model.specifications.duration
        step_ratio = self._segmentation.segmentation_step
        step = step_ratio * chunk_duration
        
        sliding_window = SlidingWindow(start=0, duration=chunk_duration, step=step)
        
        # 2. Prepare Aggregation Buffers
        # Calculate total number of output frames
        num_samples = waveform.shape[1]
        num_frames = model.num_frames(num_samples)
        
        # Map model output resolution to time
        resolution = model.receptive_field_size(1) / sample_rate # duration of 1 frame? 
        # Actually, pyannote models use a specific SlidingWindow for output
        # We rely on the model's helper to determine frame indices
        
        num_classes = len(model.specifications.classes)
        if model.specifications.powerset:
            num_classes = model.specifications.num_powerset_classes

        # Aggregation buffers
        aggregated_scores = np.zeros((num_frames, num_classes))
        aggregated_weights = np.zeros((num_frames, 1))

        # 3. Inference Loop
        chunks = sliding_window(torch.zeros(1, num_samples), sample_rate)
        
        for chunk in chunks:
            # Extract Audio
            start_sample = int(chunk.start * sample_rate)
            end_sample = int(chunk.end * sample_rate)
            
            # Handle edge case: last chunk might be shorter? 
            # Pyannote usually pads or expects fixed size. We pad audio if needed.
            wav_chunk = waveform[:, start_sample:end_sample]
            if wav_chunk.shape[1] < int(chunk_duration * sample_rate):
                pad_size = int(chunk_duration * sample_rate) - wav_chunk.shape[1]
                wav_chunk = torch.nn.functional.pad(wav_chunk, (0, pad_size))
            
            wav_chunk = wav_chunk.unsqueeze(0).to(device) # (1, C, T)

            # Extract Visual
            visual_chunk = self.load_visual_chunk(sess_name, chunk.start, chunk.end)
            visual_chunk = visual_chunk.unsqueeze(0).to(device) # (1, T, C, H, W)
            
            # Forward Pass
            inputs = {"waveforms": wav_chunk, "visual": visual_chunk}
            with torch.no_grad():
                # Shape: (1, frames_per_chunk, classes)
                prediction = model(inputs)
            
            prediction = prediction[0].cpu().numpy()
            
            # Aggregate (Overlap-Add)
            # We need to map the chunk local frames to global frames
            # Using model helper to get indices for this specific chunk
            # Note: This is a simplified calculation assuming standard convolutions
            
            # Global sample indices for this chunk
            chunk_start_sample = start_sample
            
            # Calculate where this chunk's output frames land in the global buffer
            # This logic mimics pyannote.audio.core.inference.Inference.aggregate
            
            # 1. Get number of frames for this chunk
            n_frames_chunk = prediction.shape[0]
            
            # 2. Determine start index in global buffer
            # We approximate based on the ratio num_frames / num_samples
            # A more robust way uses `model.receptive_field_center` but simple scaling is often used in scripts
            
            # Using the exact same logic as the Model.num_frames typically ensures alignment
            start_frame_global = model.num_frames(chunk_start_sample)
            
            # Safety check bounds
            end_frame_global = start_frame_global + n_frames_chunk
            
            if end_frame_global > num_frames:
                # Truncate if slight mismatch at very end
                valid_len = num_frames - start_frame_global
                prediction = prediction[:valid_len]
                end_frame_global = num_frames
            
            aggregated_scores[start_frame_global:end_frame_global] += prediction
            aggregated_weights[start_frame_global:end_frame_global] += 1.0

        # 4. Finalize
        # Avoid division by zero
        aggregated_weights[aggregated_weights == 0] = 1.0
        aggregated_scores /= aggregated_weights
        
        # Create SlidingWindowFeature
        # We need the output SlidingWindow definition from the model
        # Reconstruct it from rf info
        _, duration, step = model.get_rf_info
        # The output starts at half the receptive field duration (usually)
        # But we can approximate using the resolution we calculated earlier or standard pyannote logic
        
        output_window = SlidingWindow(start=0.0, duration=duration, step=step)
        
        return SlidingWindowFeature(aggregated_scores, output_window)

    def __call__(self, in_wav, sess_name=None):
        # Input check
        assert isinstance(in_wav, (str, ProtocolFile)), "input must be either a str or a ProtocolFile"
        in_wav_path = in_wav if not isinstance(in_wav, ProtocolFile) else in_wav['audio']
        
        print(f'Processing {sess_name}: Extracting AV segmentations.')
        
        # Load full audio
        waveform, sample_rate = torchaudio.load(in_wav_path)
        # Force mono/selected channel (SDM style)
        if waveform.shape[0] > 1:
             # Assuming channel 0 for simplicity, or mix. Original script did unsqueeze(0) on waveform[0]
             waveform = waveform[0:1, :]
        
        # --- REPLACED: Use custom AV segmentation ---
        # segmentations = self.get_segmentations(...)
        segmentations = self.get_av_segmentations(waveform, sample_rate, sess_name)
        # --------------------------------------------

        if self.apply_median_filtering:
            segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode='reflect')

        # binarize segmentation
        binarized_segmentations = segmentations

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            binarized_segmentations,
            self._segmentation.model.receptive_field_size(1) / sample_rate, # Approx duration
            warm_up=(0.0, 0.0),
        )

        print("Extracting Embeddings.")
        # Embeddings are still Audio-Only in this pipeline configuration
        # pyannote.audio SpeakerDiarizationPipeline.get_embeddings expects "waveform"
        # We pass the full waveform we loaded
        embeddings = self.get_embeddings(
            {"waveform": waveform, "sample_rate": sample_rate},
            binarized_segmentations,
            exclude_overlap=self.embedding_exclude_overlap,
        )

        # shape: (num_chunks, local_num_speakers, dimension)
        print("Clustering.")
        hard_clusters, _, _ = self.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            min_clusters=self.min_speakers,  
            max_clusters=self.max_speakers
        )

        # count capping
        count.data = np.minimum(count.data, self.max_speakers).astype(np.int8)

        # keep track of inactive speakers
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0

        # reconstruct discrete diarization
        hard_clusters[inactive_speakers] = -2
        discrete_diarization, _ = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )

        # convert to annotation
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
        "This script performs diarization using DiariZen AV pipeline",
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
        help="Path to DiariZen model hub directory."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Path to pretrained embedding model."
    )
    # Added visual argument
    parser.add_argument(
        "--visual_root",
        type=str,
        required=True,
        help="Path to directory containing video files (e.g. session_id.mp4)."
    )

    # inference parameters
    parser.add_argument("--seg_duration", type=int, default=16)
    parser.add_argument("--segmentation_step", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--apply_median_filtering", action=argparse.BooleanOptionalAction, default=True)

    # clustering parameters
    parser.add_argument("--clustering_method", type=str, default="VBxClustering", choices=["VBxClustering", "AgglomerativeClustering"])
    parser.add_argument("--min_speakers", type=int, default=1)
    parser.add_argument("--max_speakers", type=int, default=20)
    parser.add_argument("--ahc_criterion", type=str, default="distance")
    parser.add_argument("--ahc_threshold", type=float, default=0.6)
    parser.add_argument("--min_cluster_size", type=int, default=13)
    parser.add_argument("--Fa", type=float, default=0.07)
    parser.add_argument("--Fb", type=float, default=0.8)
    parser.add_argument("--lda_dim", type=int, default=128)
    parser.add_argument("--max_iters", type=int, default=20)

    # Output
    parser.add_argument("--rttm_out_dir", type=str, default=None, required=False)

    args = parser.parse_args()
    print(args)

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
    diarizen_pipeline = AV_DiariZenPipeline(
        diarizen_hub=Path(args.diarizen_hub),
        embedding_model=args.embedding_model,
        config_parse=config_parse,
        rttm_out_dir=args.rttm_out_dir,
        visual_root=args.visual_root
    )

    audio_f = scp2path(args.in_wav_scp)
    for audio_file in audio_f:
        sess_name = Path(audio_file).stem.split('.')[0]
        # In case the file has multiple extensions or complex naming, verify sess_name matches video file name
        print(f'Processing: {sess_name}')
        diarizen_pipeline(audio_file, sess_name=sess_name)