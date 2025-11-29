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
from pyannote.core import SlidingWindow, SlidingWindowFeature, Segment
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
        diarizen_hub: str = None,
        embedding_model: str = None,
        config_parse: Optional[Dict[str, Any]] = None,
        rttm_out_dir: Optional[str] = None,
        **kwargs
    ):
        self.visual_root = visual_root
        self.visual_fps = visual_fps
        self.visual_size = visual_size
        
        # --- LOGIC COPIED AND PATCHED FROM DiariZenPipeline.__init__ ---
        config_path = Path(diarizen_hub) / "config.toml"
        config = toml.load(config_path.as_posix())

        if config_parse is not None:
            print('Overriding with parsed config.')
            if "inference" not in config:
                config["inference"] = {}
            if "clustering" not in config:
                config["clustering"] = {}
            
            config["inference"]["args"] = config_parse["inference"]["args"]
            config["clustering"]["args"] = config_parse["clustering"]["args"]
       
        inference_config = config["inference"]["args"]
        clustering_config = config["clustering"]["args"]
        
        print(f'Loaded configuration: {config}')

        from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
        
        SpeakerDiarizationPipeline.__init__(
            self,
            config=config,
            seg_duration=inference_config["seg_duration"],
            segmentation=str(Path(diarizen_hub) / "pytorch_model.bin"),
            segmentation_step=inference_config["segmentation_step"],
            embedding=embedding_model,
            embedding_exclude_overlap=True,
            clustering=clustering_config["method"],     
            embedding_batch_size=inference_config["batch_size"],
            segmentation_batch_size=inference_config["batch_size"],
            device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.apply_median_filtering = inference_config["apply_median_filtering"]
        self.min_speakers = clustering_config["min_speakers"]
        self.max_speakers = clustering_config["max_speakers"]

        if clustering_config["method"] == "AgglomerativeClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": clustering_config["min_cluster_size"],
                    "threshold": clustering_config["ahc_threshold"],
                }
            }
        elif clustering_config["method"] == "VBxClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "ahc_criterion": clustering_config["ahc_criterion"],
                    "ahc_threshold": clustering_config["ahc_threshold"],
                    "Fa": clustering_config["Fa"],
                    "Fb": clustering_config["Fb"],
                }
            }
            self.clustering.plda_dir = str(Path(diarizen_hub) / "plda")
            self.clustering.lda_dim = clustering_config["lda_dim"]
            self.clustering.maxIters = clustering_config["max_iters"]
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_config['method']}")

        self.instantiate(self.PIPELINE_PARAMS)

        if rttm_out_dir is not None:
            os.makedirs(rttm_out_dir, exist_ok=True)
        self.rttm_out_dir = rttm_out_dir

        self._segmentation.model.eval()

    @classmethod
    def from_pretrained(
        cls, 
        repo_id: str, 
        visual_root: str, 
        cache_dir: str = None,
        rttm_out_dir: str = None,
    ) -> "AV_DiariZenPipeline":
        
        diarizen_hub = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )

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
        start_frame = int(start * self.visual_fps)
        end_frame = int(end * self.visual_fps)
        num_frames_target = end_frame - start_frame
        
        frames = []
        video_path = os.path.join(self.visual_root, f"{session}.mp4")
        
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
                    frames.append(np.zeros((*self.visual_size, 3), dtype=np.uint8))
            cap.release()
        else:
            frames = np.zeros((num_frames_target, *self.visual_size, 3), dtype=np.uint8)

        if not frames:
             frames = np.zeros((num_frames_target, *self.visual_size, 3), dtype=np.uint8)

        visual_tensor = torch.tensor(np.array(frames), dtype=torch.float32)
        visual_tensor = visual_tensor.permute(0, 3, 1, 2) / 255.0
        return visual_tensor

    def get_av_segmentations(self, waveform: torch.Tensor, sample_rate: int, sess_name: str) -> SlidingWindowFeature:
        model = self._segmentation.model
        device = self._segmentation.device
        
        duration = waveform.shape[1] / sample_rate
        chunk_duration = model.specifications.duration
        
        step_ratio = getattr(self._segmentation, "step", 0.1) 
        step = step_ratio * chunk_duration
        
        sliding_window = SlidingWindow(start=0, duration=chunk_duration, step=step)
        
        _, rf_duration, rf_step = model.get_rf_info
        
        num_samples = waveform.shape[1]
        num_frames = model.num_frames(num_samples)
        
        num_classes = len(model.specifications.classes)
        if model.specifications.powerset:
            num_classes = model.specifications.num_powerset_classes

        aggregated_scores = np.zeros((num_frames, num_classes))
        aggregated_weights = np.zeros((num_frames, 1))

        chunks = sliding_window(Segment(0, duration))
        
        for chunk in chunks:
            start_sample = int(chunk.start * sample_rate)
            end_sample = int(chunk.end * sample_rate)
            
            wav_chunk = waveform[:, start_sample:end_sample]
            if wav_chunk.shape[1] < int(chunk_duration * sample_rate):
                pad_size = int(chunk_duration * sample_rate) - wav_chunk.shape[1]
                wav_chunk = torch.nn.functional.pad(wav_chunk, (0, pad_size))
            
            wav_chunk = wav_chunk.unsqueeze(0).to(device) 

            visual_chunk = self.load_visual_chunk(sess_name, chunk.start, chunk.end)
            visual_chunk = visual_chunk.unsqueeze(0).to(device) 
            
            inputs = {"waveforms": wav_chunk, "visual": visual_chunk}
            with torch.no_grad():
                prediction = model(inputs)
            
            prediction = prediction[0].cpu().numpy()
            
            start_frame_global = int(np.round(chunk.start / rf_step))
            
            n_frames_chunk = prediction.shape[0]
            end_frame_global = start_frame_global + n_frames_chunk
            
            if start_frame_global < 0: 
                start_frame_global = 0
            
            if start_frame_global >= num_frames:
                continue

            if end_frame_global > num_frames:
                valid_len = num_frames - start_frame_global
                if valid_len <= 0:
                    continue
                prediction = prediction[:valid_len]
                end_frame_global = num_frames
            
            aggregated_scores[start_frame_global:end_frame_global] += prediction
            aggregated_weights[start_frame_global:end_frame_global] += 1.0

        aggregated_weights[aggregated_weights == 0] = 1.0
        aggregated_scores /= aggregated_weights
        
        output_window = SlidingWindow(start=0.0, duration=rf_duration, step=rf_step)
        
        return SlidingWindowFeature(aggregated_scores, output_window)

    def __call__(self, in_wav, sess_name=None):
        assert isinstance(in_wav, (str, ProtocolFile)), "input must be either a str or a ProtocolFile"
        in_wav_path = in_wav if not isinstance(in_wav, ProtocolFile) else in_wav['audio']
        
        print(f'Processing {sess_name}: Extracting AV segmentations.')
        
        waveform, sample_rate = torchaudio.load(in_wav_path)
        if waveform.shape[0] > 1:
             waveform = waveform[0:1, :]
        
        segmentations = self.get_av_segmentations(waveform, sample_rate, sess_name)

        if self.apply_median_filtering:
            segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode='reflect')

        binarized_segmentations = segmentations

        # --- FIX: Create SlidingWindow object for speaker_count ---
        _, rf_duration, rf_step = self._segmentation.model.get_rf_info
        frames = SlidingWindow(start=0.0, duration=rf_duration, step=rf_step)
        # ----------------------------------------------------------

        count = self.speaker_count(
            binarized_segmentations,
            frames, # Passing SlidingWindow object instead of float
            warm_up=(0.0, 0.0),
        )

        print("Extracting Embeddings.")
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