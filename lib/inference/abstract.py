"""
Abstract base class for TTS engines.
Defines the interface that all TTS implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, BinaryIO
import torch


@dataclass
class TTSConfig:
    """Configuration for TTS inference."""
    # Model settings
    model_path: str = ""
    model_type: str = "english"  # 'english' or 'multilingual'
    device: str = "cuda"
    
    # Generation parameters
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    
    # Audio settings
    sample_rate: int = 24000
    
    # Reference audio for voice cloning
    reference_audio: Optional[str] = None
    
    # Language (for multilingual model)
    language_id: str = "en"


@dataclass
class TTSResult:
    """Result of TTS generation."""
    audio_data: Union[bytes, torch.Tensor]
    sample_rate: int
    duration_sec: float
    inference_time_sec: float
    text_length: int
    
    @property
    def real_time_factor(self) -> float:
        """Calculate real-time factor (inference time / audio duration)."""
        if self.duration_sec > 0:
            return self.inference_time_sec / self.duration_sec
        return float('inf')
    
    def get_audio_bytes(self) -> bytes:
        """Get audio data as raw bytes (PCM 16-bit signed)."""
        if isinstance(self.audio_data, bytes):
            return self.audio_data
        
        # Convert tensor to bytes
        if isinstance(self.audio_data, torch.Tensor):
            audio = self.audio_data
            if audio.is_cuda:
                audio = audio.cpu()
            if audio.dim() > 1:
                audio = audio.squeeze()
            # Normalize and convert to int16
            audio = audio.float()
            if audio.abs().max() > 1.0:
                audio = audio / audio.abs().max()
            audio_int16 = (audio * 32767).to(torch.int16)
            return audio_int16.numpy().tobytes()
        
        raise TypeError(f"Unknown audio data type: {type(self.audio_data)}")


class TTSInterface(ABC):
    """
    Abstract interface for Text-to-Speech engines.
    
    All TTS implementations (local, remote, etc.) must implement this interface.
    """
    
    @abstractmethod
    def __init__(self, config: TTSConfig):
        """Initialize the TTS engine with configuration."""
        pass
    
    @abstractmethod
    def generate(self, text: str) -> TTSResult:
        """
        Generate speech from text.
        
        Args:
            text: Text to synthesize
        
        Returns:
            TTSResult containing audio data and metadata
        """
        pass
    
    @abstractmethod
    def generate_batch(self, texts: list[str]) -> list[TTSResult]:
        """
        Generate speech from multiple texts.
        
        May be optimized for batch processing on GPU.
        
        Args:
            texts: List of texts to synthesize
        
        Returns:
            List of TTSResult objects
        """
        pass
    
    @abstractmethod
    def set_reference_audio(self, audio_path: str) -> None:
        """
        Set reference audio for voice cloning.
        
        Args:
            audio_path: Path to reference audio file
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Release resources (VRAM, memory, etc.)."""
        pass
    
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Get the output sample rate."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        return False


