"""
Chatterbox client for connecting to a remote inference server.
Enables multiple ebook conversions to share a single GPU server.
"""

import time
import logging
import base64
from typing import List, Optional
from pathlib import Path

import requests

from .abstract import TTSInterface, TTSConfig, TTSResult

logger = logging.getLogger(__name__)


class ChatterboxClient(TTSInterface):
    """
    Client for remote Chatterbox inference server.
    
    Connects to a FastAPI server that handles TTS generation.
    Enables high concurrency by sharing one GPU across multiple conversion jobs.
    
    Usage:
        config = TTSConfig()
        client = ChatterboxClient(config, server_url="http://localhost:8000")
        client.set_reference_audio("/path/to/voice.wav")
        result = client.generate("Hello world")
    """
    
    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        server_url: str = "http://localhost:8000",
        timeout: int = 120,
        **kwargs
    ):
        """
        Initialize the client.
        
        Args:
            config: TTSConfig object
            server_url: URL of the inference server
            timeout: Request timeout in seconds
            **kwargs: Override config values
        """
        if config is None:
            config = TTSConfig(**kwargs)
        
        self.config = config
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self._reference_audio_b64: Optional[str] = None
        self._reference_audio_path: Optional[str] = None
        self._sr = config.sample_rate
        self._is_connected = False
        
        # Test connection
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Check if the server is reachable."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                self._is_connected = True
                data = response.json()
                self._sr = data.get('sample_rate', self._sr)
                logger.info(f"Connected to inference server at {self.server_url}")
                return True
        except requests.RequestException as e:
            logger.warning(f"Cannot connect to server: {e}")
        
        self._is_connected = False
        return False
    
    def generate(self, text: str) -> TTSResult:
        """Generate speech from text via remote server."""
        if not text.strip():
            raise ValueError("Empty text")
        
        start_time = time.time()
        
        payload = {
            "text": text,
            "exaggeration": self.config.exaggeration,
            "cfg_weight": self.config.cfg_weight,
            "temperature": self.config.temperature,
            "language_id": self.config.language_id,
        }
        
        if self._reference_audio_b64:
            payload["reference_audio_b64"] = self._reference_audio_b64
        
        try:
            response = requests.post(
                f"{self.server_url}/synthesize",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Decode audio from base64
            audio_bytes = base64.b64decode(data["audio_b64"])
            
            inference_time = time.time() - start_time
            
            return TTSResult(
                audio_data=audio_bytes,
                sample_rate=data.get("sample_rate", self._sr),
                duration_sec=data.get("duration_sec", 0),
                inference_time_sec=inference_time,
                text_length=len(text)
            )
        
        except requests.RequestException as e:
            logger.error(f"Server request failed: {e}")
            raise RuntimeError(f"Inference server error: {e}") from e
    
    def generate_batch(self, texts: List[str]) -> List[TTSResult]:
        """
        Generate speech from multiple texts.
        
        Sends a batch request to the server for efficient processing.
        """
        if not texts:
            return []
        
        # Filter empty texts
        texts = [t for t in texts if t.strip()]
        if not texts:
            return []
        
        start_time = time.time()
        
        payload = {
            "texts": texts,
            "exaggeration": self.config.exaggeration,
            "cfg_weight": self.config.cfg_weight,
            "temperature": self.config.temperature,
            "language_id": self.config.language_id,
        }
        
        if self._reference_audio_b64:
            payload["reference_audio_b64"] = self._reference_audio_b64
        
        try:
            response = requests.post(
                f"{self.server_url}/synthesize_batch",
                json=payload,
                timeout=self.timeout * len(texts)
            )
            response.raise_for_status()
            
            data = response.json()
            total_time = time.time() - start_time
            
            results = []
            for i, item in enumerate(data["results"]):
                audio_bytes = base64.b64decode(item["audio_b64"])
                results.append(TTSResult(
                    audio_data=audio_bytes,
                    sample_rate=item.get("sample_rate", self._sr),
                    duration_sec=item.get("duration_sec", 0),
                    inference_time_sec=total_time / len(texts),
                    text_length=len(texts[i])
                ))
            
            return results
        
        except requests.RequestException as e:
            logger.error(f"Batch request failed: {e}")
            # Fall back to sequential processing
            logger.info("Falling back to sequential processing")
            return [self.generate(text) for text in texts]
    
    def set_reference_audio(self, audio_path: str) -> None:
        """Set reference audio for voice cloning."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
        
        # Read and encode audio
        with open(path, 'rb') as f:
            audio_bytes = f.read()
        
        self._reference_audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        self._reference_audio_path = str(path)
        self.config.reference_audio = str(path)
        
        logger.info(f"Reference audio set: {path.name}")
    
    def cleanup(self) -> None:
        """No-op for client (server manages resources)."""
        self._reference_audio_b64 = None
        logger.info("Client cleaned up")
    
    @property
    def sample_rate(self) -> int:
        """Get the output sample rate."""
        return self._sr
    
    @property
    def is_loaded(self) -> bool:
        """Check if connected to server."""
        return self._is_connected
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to the server."""
        return self._check_connection()


