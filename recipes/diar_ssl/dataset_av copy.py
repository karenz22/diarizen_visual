import torch
import numpy as np
import cv2  # Requires opencv-python
import os
from .dataset import DiarizationDataset

class AV_DiarizationDataset(DiarizationDataset):
    def __init__(self, visual_root: str, visual_fps: int = 25, **kwargs):
        super().__init__(**kwargs)
        self.visual_root = visual_root
        self.visual_fps = visual_fps
        self.resize_dims = (224, 224) # Standard CNN input size

    def load_visual_frames(self, session, start_sec, end_sec):
        """
        Load visual frames for the specific time chunk.
        Assumes structure: visual_root/session/frame_xxxxx.jpg
        """
        start_frame = int(start_sec * self.visual_fps)
        end_frame = int(end_sec * self.visual_fps)
        
        frames = []
        # Basic loop to load frames. Optimized loaders would use video decoders directly.
        video_path = os.path.join(self.visual_root, f"{session}.mp4")
        
        if os.path.exists(video_path):
             cap = cv2.VideoCapture(video_path)
             cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
             for _ in range(end_frame - start_frame):
                 ret, frame = cap.read()
                 if ret:
                     frame = cv2.resize(frame, self.resize_dims)
                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                     frames.append(frame)
                 else:
                     # Pad with zeros if read fails
                     frames.append(np.zeros((*self.resize_dims, 3), dtype=np.uint8))
             cap.release()
        else:
            # Dummy visual data if file missing
            frames = np.zeros((end_frame - start_frame, *self.resize_dims, 3), dtype=np.uint8)

        # To Tensor (T, C, H, W) and normalize
        visual_tensor = torch.tensor(np.array(frames), dtype=torch.float32)
        visual_tensor = visual_tensor.permute(0, 3, 1, 2) / 255.0
        return visual_tensor

    def __getitem__(self, idx):
        # Get audio and labels from parent
        data, mask_label, session = super().__getitem__(idx)
        
        # Get timing info again to load visual
        _, _, chunk_start, chunk_end = self.chunk_indices[idx]
        
        visual_data = self.load_visual_frames(session, chunk_start, chunk_end)
        
        # Return dict expected by AVModel
        inputs = {
            "waveforms": data,
            "visual": visual_data
        }
        
        return inputs, mask_label, session

# Collate function needs update to handle dict input
def av_collate_fn(batch):
    collated_inputs = {"waveforms": [], "visual": []}
    collated_y = []
    collated_names = []
    
    for inputs, y, name in batch:
        collated_inputs["waveforms"].append(inputs["waveforms"])
        collated_inputs["visual"].append(inputs["visual"])
        collated_y.append(y) # simplified: assume label padding handled in dataset or similar logic
        collated_names.append(name)

    return {
        'inputs': {
            'waveforms': torch.from_numpy(np.stack(collated_inputs['waveforms'])).float(),
            'visual': torch.stack(collated_inputs['visual']).float()
        },
        'ts': torch.from_numpy(np.stack(collated_y)),
        'names': collated_names
    }