"""
Chatterbox inference server.
FastAPI server that loads the model once and serves multiple clients.

Run with:
    python -m lib.inference.server --model_path /path/to/chatterbox
    
Or:
    uvicorn lib.inference.server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import base64
import logging
import tempfile
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .local import LocalChatterboxEngine
from .abstract import TTSConfig

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global engine instance
_engine: Optional[LocalChatterboxEngine] = None


class SynthesizeRequest(BaseModel):
    """Request body for single text synthesis."""
    text: str = Field(..., min_length=1, max_length=10000)
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    language_id: str = Field(default="en", max_length=10)
    reference_audio_b64: Optional[str] = None


class SynthesizeBatchRequest(BaseModel):
    """Request body for batch text synthesis."""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    language_id: str = Field(default="en", max_length=10)
    reference_audio_b64: Optional[str] = None


class SynthesizeResponse(BaseModel):
    """Response body for synthesis."""
    audio_b64: str
    sample_rate: int
    duration_sec: float
    inference_time_sec: float


class BatchResponse(BaseModel):
    """Response body for batch synthesis."""
    results: List[SynthesizeResponse]
    total_duration_sec: float
    total_inference_time_sec: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    sample_rate: int
    device: str


def get_engine() -> LocalChatterboxEngine:
    """Get the global engine instance."""
    global _engine
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _engine


def init_engine(
    model_path: str,
    device: str = "cuda",
    model_type: str = "english"
) -> None:
    """Initialize the global engine instance."""
    global _engine
    
    logger.info(f"Initializing Chatterbox engine...")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Type: {model_type}")
    
    config = TTSConfig(
        model_path=model_path,
        device=device,
        model_type=model_type
    )
    
    _engine = LocalChatterboxEngine(config)
    logger.info("Engine initialized successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    # Startup
    model_path = os.environ.get('CHATTERBOX_MODEL_PATH', '')
    device = os.environ.get('CHATTERBOX_DEVICE', 'cuda')
    model_type = os.environ.get('CHATTERBOX_MODEL_TYPE', 'english')
    
    if model_path:
        init_engine(model_path, device, model_type)
    else:
        logger.warning("CHATTERBOX_MODEL_PATH not set - model will not be loaded")
    
    yield
    
    # Shutdown
    global _engine
    if _engine is not None:
        _engine.cleanup()
        _engine = None
    logger.info("Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Chatterbox TTS Server",
    description="TTS inference server for ebook2audiobook",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global _engine
    
    return HealthResponse(
        status="ok" if _engine is not None else "no_model",
        model_loaded=_engine is not None,
        sample_rate=_engine.sample_rate if _engine else 24000,
        device=_engine._device if _engine else "none"
    )


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """Synthesize speech from text."""
    engine = get_engine()
    
    # Handle reference audio if provided
    temp_ref = None
    if request.reference_audio_b64:
        try:
            audio_bytes = base64.b64decode(request.reference_audio_b64)
            temp_ref = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_ref.write(audio_bytes)
            temp_ref.close()
            engine.set_reference_audio(temp_ref.name)
        except Exception as e:
            logger.error(f"Failed to process reference audio: {e}")
    
    try:
        # Update generation parameters
        engine.config.exaggeration = request.exaggeration
        engine.config.cfg_weight = request.cfg_weight
        engine.config.temperature = request.temperature
        engine.config.language_id = request.language_id
        
        # Generate
        result = engine.generate(request.text)
        
        # Encode audio
        audio_bytes = result.get_audio_bytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return SynthesizeResponse(
            audio_b64=audio_b64,
            sample_rate=result.sample_rate,
            duration_sec=result.duration_sec,
            inference_time_sec=result.inference_time_sec
        )
    
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp file
        if temp_ref:
            try:
                os.unlink(temp_ref.name)
            except Exception:
                pass


@app.post("/synthesize_batch", response_model=BatchResponse)
async def synthesize_batch(request: SynthesizeBatchRequest):
    """Synthesize speech from multiple texts."""
    engine = get_engine()
    
    # Handle reference audio
    temp_ref = None
    if request.reference_audio_b64:
        try:
            audio_bytes = base64.b64decode(request.reference_audio_b64)
            temp_ref = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_ref.write(audio_bytes)
            temp_ref.close()
            engine.set_reference_audio(temp_ref.name)
        except Exception as e:
            logger.error(f"Failed to process reference audio: {e}")
    
    try:
        # Update parameters
        engine.config.exaggeration = request.exaggeration
        engine.config.cfg_weight = request.cfg_weight
        engine.config.temperature = request.temperature
        engine.config.language_id = request.language_id
        
        # Generate batch
        start_time = time.time()
        results = engine.generate_batch(request.texts)
        total_time = time.time() - start_time
        
        # Build response
        responses = []
        total_duration = 0
        
        for result in results:
            audio_bytes = result.get_audio_bytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            responses.append(SynthesizeResponse(
                audio_b64=audio_b64,
                sample_rate=result.sample_rate,
                duration_sec=result.duration_sec,
                inference_time_sec=result.inference_time_sec
            ))
            total_duration += result.duration_sec
        
        return BatchResponse(
            results=responses,
            total_duration_sec=total_duration,
            total_inference_time_sec=total_time
        )
    
    except Exception as e:
        logger.error(f"Batch synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_ref:
            try:
                os.unlink(temp_ref.name)
            except Exception:
                pass


def main():
    """Run the server from command line."""
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Chatterbox TTS Server")
    parser.add_argument("--model_path", required=True, help="Path to Chatterbox model")
    parser.add_argument("--device", default="cuda", help="Device (cuda, cpu, mps)")
    parser.add_argument("--model_type", default="turbo", choices=["english", "multilingual", "turbo"])
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    # Set environment variables for lifespan
    os.environ['CHATTERBOX_MODEL_PATH'] = args.model_path
    os.environ['CHATTERBOX_DEVICE'] = args.device
    os.environ['CHATTERBOX_MODEL_TYPE'] = args.model_type
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


