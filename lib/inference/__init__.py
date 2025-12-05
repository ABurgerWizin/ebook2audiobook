"""
Inference module for ebook2audiobook.
Provides TTS inference through local model or remote server.
"""

from .abstract import TTSInterface, TTSConfig, TTSResult
from .local import LocalChatterboxEngine
from .client import ChatterboxClient

__all__ = [
    "TTSInterface",
    "TTSConfig",
    "TTSResult",
    "LocalChatterboxEngine",
    "ChatterboxClient",
]


def get_engine(mode: str = 'local', **kwargs):
    """
    Factory function to get the appropriate TTS engine.
    
    Args:
        mode: 'local' for in-process model, 'client' for remote server
        **kwargs: Engine-specific configuration
    
    Returns:
        TTSInterface implementation
    """
    if mode == 'local':
        return LocalChatterboxEngine(**kwargs)
    elif mode == 'client':
        return ChatterboxClient(**kwargs)
    else:
        raise ValueError(f"Unknown inference mode: {mode}")


