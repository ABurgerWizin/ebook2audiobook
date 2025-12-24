#!/usr/bin/env python3
"""
Simple CLI for ebook to audiobook conversion using Chatterbox TTS.

Usage:
    python convert.py --ebook book.epub --voice voice.wav --output ./audiobooks
    python convert.py --ebook book.epub --preview  # Preview segmentation only
    python convert.py --ebook book.epub --server http://localhost:8000  # Use remote server
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert ebooks to audiobooks using Chatterbox TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion with voice cloning
  python convert.py --ebook book.epub --voice voice.wav

  # Preview segmentation without generating audio
  python convert.py --ebook book.epub --preview

  # Use remote inference server
  python convert.py --ebook book.epub --voice voice.wav --server http://localhost:8000

  # Specify output format and directory
  python convert.py --ebook book.epub --voice voice.wav --output ./audiobooks --format mp3
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--ebook", "-e",
        required=True,
        help="Path to the ebook file (epub, mobi, pdf, etc.)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--voice", "-v",
        help="Path to reference voice audio for cloning"
    )
    parser.add_argument(
        "--output", "-o",
        default="./audiobooks",
        help="Output directory (default: ./audiobooks)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["m4b", "mp3", "flac", "ogg", "wav"],
        default="m4b",
        help="Output format (default: m4b)"
    )
    
    # Preview mode
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Preview segmentation only, don't generate audio"
    )
    
    # Inference settings
    parser.add_argument(
        "--server", "-s",
        help="URL of remote inference server (enables client mode)"
    )
    parser.add_argument(
        "--model-path",
        help="Path to local Chatterbox model"
    )
    parser.add_argument(
        "--model-type",
        choices=["english", "multilingual"],
        default="english",
        help="Model type (default: english)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps"],
        default="cuda",
        help="Device for inference (default: cuda)"
    )
    parser.add_argument(
        "--max-vram",
        type=int,
        default=12,
        help="Maximum VRAM usage in GB (default: 12)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Voice exaggeration (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="CFG weight (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature (0.1-2.0, default: 0.8)"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code for multilingual model (default: en)"
    )
    
    # Processing options
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=36,
        help="Maximum tokens per segment (default: 36)"
    )
    parser.add_argument(
        "--temp-format",
        choices=["flac", "mp3", "wav"],
        default="flac",
        help="Temporary audio format (default: flac)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Don't delete temporary files after conversion"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate ebook path
    ebook_path = Path(args.ebook)
    if not ebook_path.exists():
        logger.error(f"Ebook not found: {args.ebook}")
        sys.exit(1)
    
    # Validate voice path if provided
    if args.voice:
        voice_path = Path(args.voice)
        if not voice_path.exists():
            logger.error(f"Voice file not found: {args.voice}")
            sys.exit(1)
    
    # Import pipeline (lazy import to speed up --help)
    from lib.pipeline import ConversionConfig, ConversionPipeline
    
    # Build configuration
    config = ConversionConfig(
        ebook_path=str(ebook_path),
        output_dir=args.output,
        output_format=args.format,
        voice_path=args.voice,
        inference_mode="client" if args.server else "local",
        server_url=args.server or "http://localhost:8000",
        model_path=args.model_path or "",
        model_type=args.model_type,
        device=args.device,
        max_vram=args.max_vram,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        language_id=args.language,
        max_tokens_per_batch=args.max_tokens,
        temp_format=args.temp_format,
        cleanup_temp=not args.keep_temp,
        preview_only=args.preview
    )
    
    # Create pipeline
    pipeline = ConversionPipeline(config)
    
    # Preview mode
    if args.preview:
        logger.info("Running in preview mode...")
        preview = pipeline.preview()
        print(preview)
        return
    
    # Progress callback
    def on_progress(progress):
        if progress.current_chapter:
            print(f"\r[{progress.segment_progress:.1%}] {progress.current_chapter}: "
                  f"Segment {progress.completed_segments}/{progress.total_segments} "
                  f"(ETA: {progress.estimated_remaining:.0f}s)", end="", flush=True)
    
    # Run conversion
    logger.info(f"Converting: {ebook_path.name}")
    logger.info(f"Output: {args.output}/{ebook_path.stem}.{args.format}")
    
    if args.voice:
        logger.info(f"Voice: {args.voice}")
    
    if args.server:
        logger.info(f"Using remote server: {args.server}")
    else:
        logger.info(f"Using local inference on {args.device}")
    
    try:
        output_path = pipeline.convert(progress_callback=on_progress)
        print()  # New line after progress
        logger.info(f"Audiobook created: {output_path}")
    except KeyboardInterrupt:
        print()
        logger.info("Conversion cancelled by user")
        sys.exit(1)
    except Exception as e:
        print()
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


