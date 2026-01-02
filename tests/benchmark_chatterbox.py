import time
import torch
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.inference.local import LocalChatterboxEngine, TTSConfig
from lib.conf import chatterbox_model_path, models_dir

logging.basicConfig(level=logging.INFO)

def run_benchmark(compile_model=False, use_fp16=False, warmup=False, label="Baseline"):
    print(f"\n{'='*50}")
    print(f"Running Benchmark: {label}")
    print(f"Config: compile={compile_model}, fp16={use_fp16}, warmup={warmup}")
    print(f"{'='*50}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available, benchmark might be slow/irrelevant for compilation.")
    
    config = TTSConfig(
        model_path=chatterbox_model_path,
        compile_model=compile_model,
        warmup=warmup,
        use_fp16=use_fp16,
        device=device
    )
    
    engine = None
    try:
        # Check if model path exists, if not warn
        if not os.path.exists(config.model_path):
            print(f"Model path {config.model_path} does not exist.")
            # Try to use a default or download? The engine might handle it?
            # The engine calls from_local, so it expects it to exist.
        
        engine = LocalChatterboxEngine(config)
        
        # Check if we need reference audio
        # If the model has built-in conds, we don't need it.
        # But for robustness, we might want to check.
        # We can't easily check internal state here without accessing protected members.
        # We'll just try generation.
        
        text = "The quick brown fox jumps over the lazy dog. This is a benchmark for speed."
        
        times = []
        rtfs = []
        
        for i in range(1, 4):
            print(f"\nGeneration {i}...")
            start = time.time()
            try:
                res = engine.generate(text)
                dur = time.time() - start
                times.append(dur)
                rtfs.append(res.real_time_factor)
                print(f"  Duration: {dur:.4f}s")
                print(f"  RTF:      {res.real_time_factor:.4f}")
                print(f"  Audio:    {len(res.audio_data) if hasattr(res.audio_data, '__len__') else 'N/A'} samples")
            except Exception as e:
                print(f"  Generation failed: {e}")
                if "conds" in str(e) or "reference" in str(e).lower():
                     print("  (It seems a reference voice is required but not provided)")
                break

        if times:
            avg_time = sum(times) / len(times)
            avg_rtf = sum(rtfs) / len(rtfs)
            print(f"\nAverage Duration: {avg_time:.4f}s")
            print(f"Average RTF:      {avg_rtf:.4f}")

    except Exception as e:
        print(f"Benchmark failed during init: {e}")
    finally:
        if engine:
            engine.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Starting Benchmark...")
    
    # Baseline
    run_benchmark(compile_model=False, use_fp16=False, warmup=False, label="Baseline (FP32 Eager)")
    
    # Optimized
    run_benchmark(compile_model=True, use_fp16=True, warmup=True, label="Optimized (FP16 Compiled)")

