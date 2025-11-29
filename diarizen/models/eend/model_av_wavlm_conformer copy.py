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
        # Standard ResNet-like stem or simple CNN stack
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
            
            # Global Average Pooling per frame
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(256, output_dim)

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, Channels, Height, Width)
        Returns:
            x: (Batch, Time, OutputDim)
        """
        if x.dim() != 5:
            raise ValueError(f"Visual input must be 5D (B, T, C, H, W), got {x.shape}")

        batch, time, C, H, W = x.shape
        # Flatten batch and time to pass through 2D CNN
        x = x.view(batch * time, C, H, W)
        
        x = self.conv_blocks(x) # (B*T, 256, 1, 1)
        x = x.flatten(1)        # (B*T, 256)
        x = self.proj(x)        # (B*T, output_dim)
        
        # Reshape back to sequence
        x = x.view(batch, time, -1)
        return x

class AVModel(BaseModel):
    def __init__(
        self,
        wavlm_src: str = "wavlm_base",
        wavlm_layer_num: int = 13,
        wavlm_feat_dim: int = 768,
        visual_input_dim: int = 256, # Dimension after visual encoder projection
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

        # --- Audio Branch (WavLM) ---
        self.wavlm_model = self.load_wavlm(wavlm_src)
        self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)
        self.audio_proj = nn.Linear(wavlm_feat_dim, attention_in)
        self.lnorm_audio = nn.LayerNorm(attention_in)

        # --- Visual Branch ---
        self.visual_encoder = VisualEncoder(output_dim=attention_in)
        self.lnorm_visual = nn.LayerNorm(attention_in)

        # --- Fusion ---
        # Concatenate Audio (attn_in) + Visual (attn_in) -> 2 * attn_in
        self.fusion_proj = nn.Linear(attention_in * 2, attention_in)
        self.lnorm_fusion = nn.LayerNorm(attention_in)

        # --- Backend (Conformer) ---
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

    def load_wavlm(self, source: str):
        """
        Load a WavLM model from either a config name or a checkpoint file.
        """
        if os.path.isfile(source):
            # Load from checkpoint file
            ckpt = torch.load(source, map_location="cpu")

            if "config" not in ckpt or "state_dict" not in ckpt:
                raise ValueError("Checkpoint must contain 'config' and 'state_dict'.")

            for k, v in ckpt["config"].items():
                if 'prune' in k and v is not False:
                    raise ValueError(f"Pruning must be disabled. Found: {k}={v}")

            model = wavlm_model(**ckpt["config"])
            model.load_state_dict(ckpt["state_dict"], strict=False)

        else:
            # Load from predefined config
            config = get_config(source)
            model = wavlm_model(**config)

        return model

    def wav2wavlm(self, in_wav, model):
        """
        transform wav to wavlm features
        """
        layer_reps, _ = model.extract_features(in_wav)
        return torch.stack(layer_reps, dim=-1)

    def forward(self, inputs) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        inputs : dict or torch.Tensor
            If dict:
                {'waveforms': (batch, channel, sample), 'visual': (batch, time, C, H, W)}
            If Tensor (legacy/audio-only):
                (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        # 1. Unpack Inputs
        if isinstance(inputs, torch.Tensor):
            waveforms = inputs
            visual = None
        elif isinstance(inputs, dict):
            waveforms = inputs.get("waveforms")
            visual = inputs.get("visual")
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")

        # 2. Process Audio
        assert waveforms.dim() == 3
        waveforms = waveforms[:, self.selected_channel, :]

        wavlm_feat = self.wav2wavlm(waveforms, self.wavlm_model)
        wavlm_feat = self.weight_sum(wavlm_feat)
        wavlm_feat = torch.squeeze(wavlm_feat, -1)  # (B, T_aud, D_wavlm)
        
        audio_emb = self.audio_proj(wavlm_feat)     # (B, T_aud, attn_in)
        audio_emb = self.lnorm_audio(audio_emb)

        # 3. Process Visual & Fusion
        if visual is not None:
            visual_emb = self.visual_encoder(visual) # (B, T_vis, attn_in)
            visual_emb = self.lnorm_visual(visual_emb)
            
            # Align Visual (T_vis) to Audio (T_aud) via interpolation
            # Transpose to (B, C, T) for interpolate
            visual_emb = visual_emb.transpose(1, 2) 
            
            # Interpolate to match audio length
            if visual_emb.shape[2] != audio_emb.shape[1]:
                visual_emb = nn.functional.interpolate(
                    visual_emb, 
                    size=audio_emb.shape[1], 
                    mode='linear', 
                    align_corners=False
                )
            
            visual_emb = visual_emb.transpose(1, 2) # Back to (B, T_aud, C)
            
            # Concatenate and Project
            fused = torch.cat([audio_emb, visual_emb], dim=-1)
            x = self.fusion_proj(fused)
            x = self.lnorm_fusion(x)
        else:
            # Fallback (Audio Only)
            x = audio_emb

        # 4. Backend & Classification
        x = self.conformer(x)
        x = self.classifier(x)
        x = self.activation(x)

        return x

    # =========================================================================
    # Required Pyannote.Audio Properties (Copied from model_wavlm_conformer.py)
    # =========================================================================

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """
        # These constants define the effective downsampling of the WavLM+Conformer stack
        kernel_size = [10, 3, 3, 3, 3, 2, 2]
        stride = [5, 2, 2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1, 1]

        return multi_conv_num_frames(
            num_samples,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """
        kernel_size = [10, 3, 3, 3, 3, 2, 2]
        stride = [5, 2, 2, 2, 2, 2, 2]
        dilation = [1, 1, 1, 1, 1, 1, 1]

        return multi_conv_receptive_field_size(
            num_frames,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """
        kernel_size = [10, 3, 3, 3, 3, 2, 2]
        stride = [5, 2, 2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1, 1]

        return multi_conv_receptive_field_center(
            frame,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    
    @property
    def get_rf_info(self):     
        """Return receptive field info to dataset
        """
        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        num_frames = self.num_frames(self.chunk_size * self.sample_rate)
        duration = receptive_field_size / self.sample_rate
        step = receptive_field_step / self.sample_rate
        return num_frames, duration, step

if __name__ == '__main__':
    # Simple test
    wavlm_conf_name = 'wavlm_base_md_s80'
    try:
        model = AVModel(wavlm_src=wavlm_conf_name)
        print("Model initialized successfully.")
        print(model)
        
        # Fake Inputs
        # Audio: Batch=2, Channel=1, Samples=32000
        audio = torch.randn(2, 1, 32000)
        # Visual: Batch=2, Time=25, Channels=3, H=224, W=224 (e.g. 25fps for 1 sec)
        visual = torch.randn(2, 25, 3, 224, 224)
        
        inputs = {"waveforms": audio, "visual": visual}
        
        # Mock specifications for 'dimension' property
        from pyannote.audio.core.task import Specifications
        model.specifications = Specifications(
            problem=None, resolution=None, duration=None, classes=["spk1", "spk2"]
        )
        
        y = model(inputs)
        print(f'Output shape: {y.shape}')
    except Exception as e:
        print(f"Test failed (likely due to missing config/checkpoint in test env): {e}")