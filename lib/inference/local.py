"""
Local Chatterbox TTS engine implementation.
Loads the model in-process for single-user inference.
"""

import gc
import time
import logging
import os
import sys
import contextlib
from typing import Optional, List
from pathlib import Path

import torch
import torchaudio

from .abstract import TTSInterface, TTSConfig, TTSResult

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_output():
    """Suppress stdout and stderr to hide underlying library progress bars."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class LocalChatterboxEngine(TTSInterface):
    """
    Local Chatterbox TTS engine.
    
    Loads the Chatterbox model directly into GPU/CPU memory.
    Best for single-user scenarios or when a remote server is not available.
    
    Usage:
        config = TTSConfig(model_path="/path/to/chatterbox")
        engine = LocalChatterboxEngine(config)
        engine.set_reference_audio("/path/to/voice.wav")
        result = engine.generate("Hello world")
        # Use result.audio_data or result.get_audio_bytes()
        engine.cleanup()
    """
    
    def __init__(self, config: Optional[TTSConfig] = None, **kwargs):
        """
        Initialize the local Chatterbox engine.
        
        Args:
            config: TTSConfig object or None to use kwargs
            **kwargs: Override config values
        """
        if config is None:
            config = TTSConfig(**kwargs)
        else:
            # Apply kwargs overrides
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.config = config
        self._model = None
        self._model_type = config.model_type
        self._device = self._validate_device(config.device)
        self._reference_audio = config.reference_audio
        self._sr = config.sample_rate
        
        # Load model
        self._load_model()
    
    def _validate_device(self, device: str) -> str:
        """Validate and normalize device specification."""
        if device == "cpu":
            return "cpu"
        
        if device == "cuda" or device.startswith("cuda:"):
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
            
            if device == "cuda":
                return "cuda"
            
            try:
                gpu_id = int(device.split(":")[1])
                if gpu_id >= torch.cuda.device_count():
                    logger.warning(f"GPU {gpu_id} not available, using cuda:0")
                    return "cuda:0"
                return device
            except (IndexError, ValueError):
                return "cuda:0"
        
        if device == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        
        logger.warning(f"Unknown device '{device}', falling back to CPU")
        return "cpu"
    
    def _load_model(self):
        """Load the Chatterbox model."""
        logging.getLogger("ChatterTTS").setLevel(logging.WARNING)
        logging.getLogger("chatterbox").setLevel(logging.WARNING)

        logger.info(f"Loading Chatterbox [{self._model_type}] on {self._device}")
        
        start_time = time.time()
        
        try:
            logger.info(f"Model path: {self.config.model_path}")
            
            with suppress_output():
                if self._model_type == "english":   
                    from chatterbox.tts import ChatterboxTTS
                    self._model = ChatterboxTTS.from_local(
                        ckpt_dir=self.config.model_path,
                        device=self._device
                    )
                elif self._model_type == "turbo":
                    from chatterbox.tts_turbo import ChatterboxTurboTTS
                    self._model = ChatterboxTurboTTS.from_local(
                        ckpt_dir=self.config.model_path,
                        device=self._device
                    )
                else:
                    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                    self._model = ChatterboxMultilingualTTS.from_local(
                        ckpt_dir=self.config.model_path,
                        device=self._device
                    )
            
            self._sr = self._model.sr
            load_time = time.time() - start_time
            
            logger.info(f"Model loaded in {load_time:.2f}s")
            logger.info(f"Sample rate: {self._sr} Hz")
            
            if self._device.startswith("cuda"):
                vram_gb = torch.cuda.memory_allocated(self._device) / 1024**3
                logger.info(f"VRAM allocated: {vram_gb:.2f} GB")
        
        except ImportError as e:
            raise ImportError(
                "Chatterbox not installed. Install with: pip install chatterbox-tts"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, text: str) -> TTSResult:
        """Generate speech from text."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        if not text.strip():
            raise ValueError("Empty text")
        
        start_time = time.time()
        
        try:
            with suppress_output():
                if self._reference_audio:
                    wav = self._model.generate(
                        text,
                        audio_prompt_path=self._reference_audio,
                        exaggeration=self.config.exaggeration,
                        cfg_weight=self.config.cfg_weight
                    )
                else:
                    if self._model_type == "multilingual":
                        wav = self._model.generate(
                            text,
                            language_id=self.config.language_id
                        )
                    else:
                        wav = self._model.generate(text)
            
            inference_time = time.time() - start_time
            
            # Ensure wav is on CPU for further processing
            if wav.is_cuda:
                wav = wav.cpu()
            
            # Calculate duration
            duration = wav.shape[-1] / self._sr
            
            return TTSResult(
                audio_data=wav,
                sample_rate=self._sr,
                duration_sec=duration,
                inference_time_sec=inference_time,
                text_length=len(text)
            )
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_batch(self, texts: List[str]) -> List[TTSResult]:
        """
        Generate speech from multiple texts.
        
        Note: Chatterbox doesn't have native batching for different texts,
        so this processes sequentially. For true batching, use chatterbox-vllm.
        """
        results = []
        for text in texts:
            if text.strip():
                results.append(self.generate(text))
        return results
    
    def set_reference_audio(self, audio_path: str) -> None:
        """Set reference audio for voice cloning."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
        
        # Validate audio format
        try:
            info = torchaudio.info(str(path))
            duration = info.num_frames / info.sample_rate
            logger.info(f"Reference audio: {path.name} ({duration:.1f}s, {info.sample_rate}Hz)")
            
            if duration < 3.0:
                logger.warning("Reference audio is very short (<3s), quality may suffer")
            elif duration > 30.0:
                logger.warning("Reference audio is long (>30s), using first 30s")
        except Exception as e:
            logger.warning(f"Could not validate reference audio: {e}")
        
        self._reference_audio = str(path)
        self.config.reference_audio = str(path)
    
    def cleanup(self) -> None:
        """Release model and VRAM."""
        if self._model is None:
            return
        
        vram_before = 0
        if self._device.startswith("cuda"):
            vram_before = torch.cuda.memory_allocated(self._device) / 1024**3
        
        del self._model
        self._model = None
        
        gc.collect()
        
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated(self._device) / 1024**3
            freed = vram_before - vram_after
            logger.info(f"Model cleaned up: {freed:.2f} GB VRAM freed")
        else:
            logger.info("Model cleaned up from memory")
    
    @property
    def sample_rate(self) -> int:
        """Get the output sample rate."""
        return self._sr
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
    
    def reload(self) -> None:
        """Reload the model after cleanup."""
        if self.is_loaded:
            logger.info("Model already loaded")
            return
        self._load_model()


