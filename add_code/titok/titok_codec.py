"""TiTok Codec for image tokenization and reconstruction.

This module provides a TiTokCodec class that wraps the TiTok model for encoding
images into discrete tokens and decoding tokens back to images. It supports batch
processing with optional AMP (Automatic Mixed Precision) for efficient inference.

Typical usage:
    codec = TiTokCodec(ckpt_dir="path/to/checkpoint", device="cuda")
    reconstructed_images = codec.decode_tokens(tokens)
"""

import os
import json
from pathlib import Path
from typing import Union, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

from modeling.titok import TiTok

class TiTokCodec:
    
    def __init__(
        self,
        ckpt_dir: Union[str, Path],
        device: str = "cuda",
        batch_size: int = 32,
        num_workers: int = 8,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
    ):

        self.ckpt_dir = Path(ckpt_dir)
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        
        self._load_model()
        
    def _load_model(self):
        config_path = self.ckpt_dir / "config.yaml"
        bin_path = self.ckpt_dir / "pytorch_model.bin"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not bin_path.exists():
            raise FileNotFoundError(f"Model file not found: {bin_path}")
        
        self.config = OmegaConf.load(str(config_path))
        self.model = TiTok(self.config)
        self.model.load_pretrained_weight(str(bin_path))
        self.model.eval()
        self.model.requires_grad_(False)
        self.model = self.model.to(self.device)
        
        assert getattr(self.model, "quantize_mode", None) == "vq", \
            "Only VQ-mode TiTok tokenizer is supported"
        
        self.target_size = self._infer_target_size()
        
        print(f"[TiTokCodec] Model loaded from {self.ckpt_dir}")
        print(f"[TiTokCodec] Target size: {self.target_size}")
        print(f"[TiTokCodec] Device: {self.device}, AMP: {self.use_amp}")
    
    def _infer_target_size(self) -> Optional[Tuple[int, int]]:
        for key in ["image_size", "img_size", "resolution"]:
            try:
                v = OmegaConf.select(self.config, f"model.{key}")
            except Exception:
                v = None
            if v is None:
                continue
            if isinstance(v, (list, tuple)) and len(v) == 2:
                return (int(v[0]), int(v[1]))
            if isinstance(v, int):
                return (int(v), int(v))
        return None
    
    
    @torch.no_grad()
    def decode_tokens(
        self,
        tokens: torch.Tensor,
        return_pil: bool = True
    ) -> Union[List[Image.Image], torch.Tensor]:

        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)  # Add channel dimension
        
        assert tokens.dim() == 3 and tokens.shape[1] == 1, \
            f"Expected tokens shape (N, 1, 32), got {tokens.shape}"
        
        tokens = tokens.to(self.device)
        
        N = tokens.shape[0]
        batch_size = self.batch_size
        
        all_images = []
        
        for i in range(0, N, batch_size):
            batch_tokens = tokens[i:i+batch_size]
            
            if self.use_amp:
                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=True):
                    recon = self.model.decode_tokens(batch_tokens)
            else:
                recon = self.model.decode_tokens(batch_tokens)
            
            recon = torch.clamp(recon, 0.0, 1.0)
            all_images.append(recon.cpu())
        
        all_images = torch.cat(all_images, dim=0)  # Shape: (N, 3, H, W)
        
        if return_pil:
            pil_images = []
            for i in range(N):
                img_tensor = (all_images[i] * 255.0).permute(1, 2, 0).to(torch.uint8).numpy()
                pil_images.append(Image.fromarray(img_tensor))
            return pil_images
        else:
            return all_images
    