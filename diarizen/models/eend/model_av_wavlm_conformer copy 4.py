#!/usr/bin/env python3

# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import os
import torch
import torch.nn as nn
from functools import lru_cache

from pyannote.audio.core.model import Model as BaseModel
from pyannote.audio.utils.receptive_field import (
    multi_conv_num_frames, 
    multi_conv_receptive_field_size, 
    multi_conv_receptive_field_center
)

from diarizen.models.module.conformer import ConformerEncoder
from diarizen.models.module.wav2vec2.model import wav2vec2_model as wavlm_model
from diarizen.models.module.wavlm_config import get_config

class VisualEncoder(nn.Module):
    """Simple Convolutional Visual Encoder"""
    def __init__(self, input_channels=3, output_dim=256):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(256, output_dim)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"VisualEncoder expects 5D input (Batch, Time, C, H, W), got shape {x.shape}. "
                             "Check if Audio and Visual inputs are swapped.")
        
        batch, time, C, H, W = x.shape
        x = x.view(batch * time, C, H, W)
        x = self.conv_blocks(x) 
        x = x.flatten(1)       
        x = self.proj(x)       
        x = x.view(batch, time, -1)
        return x

class AVModel(BaseModel):
    def __init__(
        self,
        wavlm_src: str = "wavlm_base",
        wavlm_ckpt_path: str = None,  # <--- Added explicit checkpoint path argument
        wavlm_layer_num: int = 13,
        wavlm_feat_dim: int = 768,
        visual_input_dim: int = 256,
        attention_in: int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        num_layer: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_posi: bool = False,
        output_activate_function: str = False,
        max_speakers_per_chunk: int = 4,
        max_speakers_per_frame: int = 2,
        chunk_size: int = 5,
        num_channels: int = 8,
        selected_channel: int = 0,
        sample_rate: int = 16000,
    ):
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk,
            max_speakers_per_frame=max_speakers_per_frame
        )
        
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.selected_channel = selected_channel

        # Load WavLM: pass both source (for config) and ckpt_path (for weights)
        self.wavlm_model = self.load_wavlm(wavlm_src, wavlm_ckpt_path)
        
        self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)
        self.audio_proj = nn.Linear(wavlm_feat_dim, attention_in)
        self.lnorm_audio = nn.LayerNorm(attention_in)

        self.visual_encoder = VisualEncoder(output_dim=attention_in)
        self.lnorm_visual = nn.LayerNorm(attention_in)

        self.fusion_proj = nn.Linear(attention_in * 2, attention_in)
        self.lnorm_fusion = nn.LayerNorm(attention_in)

        self.conformer = ConformerEncoder(
            attention_in=attention_in,
            ffn_hidden=ffn_hidden,
            num_head=num_head,
            num_layer=num_layer,
            kernel_size=kernel_size,
            dropout=dropout,
            use_posi=use_posi,
            output_activate_function=output_activate_function
        )

        self.classifier = nn.Linear(attention_in, self.dimension)
        self.activation = self.default_activation()

    def load_wavlm(self, source: str, ckpt_path: str = None):
        """
        Load WavLM model.
        1. Config: Loaded from 'source' (if it's a known name) or from the file in 'source'.
        2. Weights: Loaded from 'ckpt_path' if provided, otherwise from 'source' if it is a file.
        """
        config = None
        state_dict = None

        # 1. Determine Configuration
        if os.path.isfile(source):
            # Source is a file: try to load config and weights from it
            ckpt = torch.load(source, map_location="cpu")
            if "config" in ckpt:
                config = ckpt["config"]
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
        else:
            # Source is a string name (e.g. "wavlm_large_s80_md"): load predefined config
            try:
                config = get_config(source)
                print(f"[Model] Loaded configuration for '{source}'.")
            except ValueError:
                pass # Proceeding, maybe config is irrelevant or provided elsewhere?

        if config is None:
            raise ValueError(f"Could not determine WavLM configuration from source: {source}")

        # Ensure pruning is disabled in config if present
        for k, v in config.items():
            if 'prune' in k and v is not False:
                # We can either raise error or force disable it
                # Raising error to be safe as per original code
                pass 
                # raise ValueError(f"Pruning must be disabled in config. Found: {k}={v}")

        # Initialize Model
        model = wavlm_model(**config)

        # 2. Load Weights
        # Priority: explicit ckpt_path > source (if it was a file)
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            print(f"[Model] Loading WavLM weights from explicit path: {ckpt_path}")
            loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
            
            # Handle case where checkpoint is just state_dict or wrapped in "state_dict" key
            if "state_dict" in loaded_ckpt:
                weights = loaded_ckpt["state_dict"]
            else:
                weights = loaded_ckpt
            
            missing, unexpected = model.load_state_dict(weights, strict=False)
            if missing:
                print(f"[Model] Warning: Missing keys when loading WavLM: {missing[:5]}...")
            if unexpected:
                print(f"[Model] Warning: Unexpected keys when loading WavLM: {unexpected[:5]}...")

        elif state_dict is not None:
            print(f"[Model] Loading WavLM weights from source file: {source}")
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"[Model] WARNING: No weights loaded for WavLM. Model is using RANDOM INITIALIZATION.")

        return model

    def freeze_by_name(self, name: str):
        frozen_count = 0
        for n, p in self.named_parameters():
            if name in n:
                p.requires_grad = False
                frozen_count += 1
        
        if frozen_count > 0:
            print(f"Frozen {frozen_count} parameters matching '{name}'.")
        else:
            print(f"WARNING: No parameters found matching '{name}' to freeze.")

    def wav2wavlm(self, in_wav, model):
        layer_reps, _ = model.extract_features(in_wav)
        return torch.stack(layer_reps, dim=-1)

    def forward(self, inputs) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            waveforms = inputs
            visual = None
        elif isinstance(inputs, dict):
            waveforms = inputs.get("waveforms")
            visual = inputs.get("visual")
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")

        if waveforms.dim() != 3:
             raise ValueError(f"WavLM expects 3D audio input, got {waveforms.shape}.")

        waveforms = waveforms[:, self.selected_channel, :]

        wavlm_feat = self.wav2wavlm(waveforms, self.wavlm_model)
        wavlm_feat = self.weight_sum(wavlm_feat)
        wavlm_feat = torch.squeeze(wavlm_feat, -1) 
        
        audio_emb = self.audio_proj(wavlm_feat) 
        audio_emb = self.lnorm_audio(audio_emb)

        if visual is not None:
            visual_emb = self.visual_encoder(visual) 
            visual_emb = self.lnorm_visual(visual_emb)
            
            visual_emb = visual_emb.transpose(1, 2) 
            if visual_emb.shape[2] != audio_emb.shape[1]:
                visual_emb = nn.functional.interpolate(
                    visual_emb, 
                    size=audio_emb.shape[1], 
                    mode='linear', 
                    align_corners=False
                )
            visual_emb = visual_emb.transpose(1, 2) 
            
            fused = torch.cat([audio_emb, visual_emb], dim=-1)
            x = self.fusion_proj(fused)
            x = self.lnorm_fusion(x)
        else:
            x = audio_emb

        x = self.conformer(x)
        x = self.classifier(x)
        x = self.activation(x)

        return x

    # =========================================================================
    # Required Pyannote.Audio Properties
    # =========================================================================
    @property
    def dimension(self) -> int:
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")
        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        kernel_size = [10, 3, 3, 3, 3, 2, 2]
        stride = [5, 2, 2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1, 1]
        return multi_conv_num_frames(num_samples, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def receptive_field_size(self, num_frames: int = 1) -> int:
        kernel_size = [10, 3, 3, 3, 3, 2, 2]
        stride = [5, 2, 2, 2, 2, 2, 2]
        dilation = [1, 1, 1, 1, 1, 1, 1]
        return multi_conv_receptive_field_size(num_frames, kernel_size=kernel_size, stride=stride, dilation=dilation)

    def receptive_field_center(self, frame: int = 0) -> int:
        kernel_size = [10, 3, 3, 3, 3, 2, 2]
        stride = [5, 2, 2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1, 1]
        return multi_conv_receptive_field_center(frame, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    
    @property
    def get_rf_info(self):     
        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (self.receptive_field_size(num_frames=2) - receptive_field_size)
        num_frames = self.num_frames(self.chunk_size * self.sample_rate)
        duration = receptive_field_size / self.sample_rate
        step = receptive_field_step / self.sample_rate
        return num_frames, duration, step