#!/usr/bin/env python3
"""
ebook2audiobook - Convert ebooks to audiobooks using Chatterbox TTS.

Usage:
    # GUI mode (default)
    ./ebook2audiobook.sh
    
    # Headless CLI mode
    ./ebook2audiobook.sh --headless --ebook /path/to/book.epub
    
    # Preview segmentation (dry-run)
    ./ebook2audiobook.sh --headless --ebook /path/to/book.epub --preview
"""

import argparse
import logging
import multiprocessing
import os
import socket
import sys
from pathlib import Path

from lib.conf import (
    NATIVE, FULL_DOCKER, prog_version, min_python_version, max_python_version,
    interface_host, interface_port, interface_concurrency_limit, max_upload_size,
    debug_mode, audiobooks_cli_dir, chatterbox_model_path, chatterbox_defaults,
    default_output_format, voices_dir, ebook_formats, devices
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def init_multiprocessing():
    """Configure multiprocessing start method."""
    method = "fork" if sys.platform in ("darwin", "linux") else "spawn"
    try:
        multiprocessing.set_start_method(method)
    except RuntimeError:
        pass


def check_python_version() -> bool:
    """Verify Python version is within supported range."""
    current = sys.version_info[:2]
    if current < min_python_version or current > max_python_version:
        logger.error(
            f"Python {current[0]}.{current[1]} not supported. "
            f"Requires {min_python_version[0]}.{min_python_version[1]} - "
            f"{max_python_version[0]}.{max_python_version[1]}"
        )
        return False
    return True


def is_port_in_use(port: int) -> bool:
    """Check if a port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('0.0.0.0', port)) == 0


def detect_device() -> str:
    """Detect the best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser with all supported options."""
    parser = argparse.ArgumentParser(
        description='Convert eBooks to Audiobooks using Chatterbox TTS.',
        epilog='''
Examples:
    # Launch GUI
    ./ebook2audiobook.sh
    
    # Convert single ebook (headless)
    ./ebook2audiobook.sh --headless --ebook book.epub --voice voice.wav
    
    # Convert directory of ebooks
    ./ebook2audiobook.sh --headless --ebooks_dir ./books/ --output_dir ./audiobooks/
    
    # Preview segmentation without conversion
    ./ebook2audiobook.sh --headless --ebook book.epub --preview
        ''',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Mode selection
    mode_group = parser.add_argument_group('Mode')
    mode_group.add_argument('--headless', action='store_true',
        help='Run in headless CLI mode (no GUI)')
    mode_group.add_argument('--preview', '--dry-run', action='store_true', dest='preview',
        help='Preview segmentation without generating audio')
    mode_group.add_argument('--preview-segments', type=int, default=16,
        help='Number of segments per chapter to preview (default: 16)')
    mode_group.add_argument('--share', action='store_true',
        help='Create a public Gradio share link')
    
    # Input/Output
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--ebook', type=str,
        help='Path to ebook file for conversion')
    io_group.add_argument('--ebooks_dir', type=str,
        help='Directory containing ebooks to convert')
    io_group.add_argument('--output_dir', type=str, default=audiobooks_cli_dir,
        help=f'Output directory (default: {audiobooks_cli_dir})')
    io_group.add_argument('--output_format', type=str, default=default_output_format,
        choices=['m4b', 'mp3', 'flac', 'wav', 'ogg'],
        help=f'Output audio format (default: {default_output_format})')
    
    # Voice/Model
    voice_group = parser.add_argument_group('Voice & Model')
    voice_group.add_argument('--voice', type=str,
        help='Path to voice reference audio for cloning')
    voice_group.add_argument('--custom_model', type=str,
        help='Path to custom Chatterbox model directory')
    voice_group.add_argument('--model_type', type=str, default='turbo',
        choices=['english', 'multilingual', 'turbo'],
        help='Chatterbox model type (default: turbo)')
    voice_group.add_argument('--device', type=str, default=None,
        choices=['cpu', 'cuda', 'mps'],
        help='Compute device (auto-detected if not specified)')
    voice_group.add_argument('--max_vram', type=int, default=12,
        help='Maximum VRAM usage in GB (default: 12)')
    
    # Chatterbox generation parameters
    gen_group = parser.add_argument_group('Generation Parameters')
    gen_group.add_argument('--exaggeration', type=float, 
        default=chatterbox_defaults['exaggeration'],
        help=f"Voice exaggeration factor (default: {chatterbox_defaults['exaggeration']})")
    gen_group.add_argument('--cfg_weight', type=float,
        default=chatterbox_defaults['cfg_weight'],
        help=f"CFG weight for generation (default: {chatterbox_defaults['cfg_weight']})")
    gen_group.add_argument('--temperature', type=float,
        default=chatterbox_defaults['temperature'],
        help=f"Sampling temperature (default: {chatterbox_defaults['temperature']})")
    gen_group.add_argument('--language', type=str, default='en',
        help='Language code for multilingual model (default: en)')
    gen_group.add_argument('--max_tokens', type=int, default=106,
        help='Maximum tokens per batch (default: 100)')
    
    # Legacy/Deprecated (kept for compatibility)
    legacy_group = parser.add_argument_group('Legacy Options (deprecated)')
    legacy_group.add_argument('--speed', type=float, default=1.0,
        help='[DEPRECATED] Speed adjustment not supported by Chatterbox')
    
    # Internal
    parser.add_argument('--script_mode', type=str, default=NATIVE,
        help=argparse.SUPPRESS)
    parser.add_argument('--version', action='version',
        version=f'ebook2audiobook v{prog_version}')
    
    return parser


def run_headless(args: argparse.Namespace) -> int:
    """Run conversion in headless (CLI) mode."""
    from lib.pipeline import ConversionPipeline, ConversionConfig, preview_segmentation
    
    # Validate input
    if not args.ebook and not args.ebooks_dir:
        logger.error("Headless mode requires --ebook or --ebooks_dir")
        return 1
    
    if args.ebook and args.ebooks_dir:
        logger.error("Cannot specify both --ebook and --ebooks_dir")
        return 1
    
    # Warn about deprecated options
    if args.speed != 1.0:
        logger.warning("--speed is deprecated and has no effect with Chatterbox TTS")
    
    # Collect ebooks to process
    ebook_paths = []
    if args.ebook:
        ebook_path = Path(args.ebook).resolve()
        if not ebook_path.exists():
            logger.error(f"Ebook not found: {ebook_path}")
            return 1
        ebook_paths.append(ebook_path)
    else:
        ebooks_dir = Path(args.ebooks_dir).resolve()
        if not ebooks_dir.is_dir():
            logger.error(f"Directory not found: {ebooks_dir}")
            return 1
        for ext in ebook_formats:
            ebook_paths.extend(ebooks_dir.glob(f"*{ext}"))
        if not ebook_paths:
            logger.error(f"No ebooks found in {ebooks_dir}")
            return 1
        logger.info(f"Found {len(ebook_paths)} ebooks to process")
    
    # Validate voice file if specified
    voice_path = None
    if args.voice:
        voice_path = Path(args.voice).resolve()
        if not voice_path.exists():
            logger.error(f"Voice file not found: {voice_path}")
            return 1
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine model path
    model_path = args.custom_model if args.custom_model else chatterbox_model_path
    
    # Preview mode - just show segmentation
    if args.preview:
        for ebook_path in ebook_paths:
            logger.info(f"Previewing: {ebook_path.name}")
            preview = preview_segmentation(str(ebook_path), preview_segments=args.preview_segments)
            print(preview)
        return 0
    
    # Device selection
    device = args.device if args.device else detect_device()
    logger.info(f"Using device: {device}")
    
    # Process each ebook
    failed = []
    for ebook_path in ebook_paths:
        logger.info(f"Converting: {ebook_path.name}")
        
        config = ConversionConfig(
            ebook_path=str(ebook_path),
            output_dir=str(output_dir),
            output_format=args.output_format,
            voice_path=str(voice_path) if voice_path else None,
            model_path=model_path,
            model_type=args.model_type,
            device=device,
            max_vram=args.max_vram,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            language_id=args.language,
            max_tokens_per_batch=args.max_tokens
        )
        
        try:
            pipeline = ConversionPipeline(config)
            
            def progress_callback(progress):
                pct = progress.segment_progress * 100
                logger.info(
                    f"[{progress.current_chapter}] "
                    f"{progress.completed_segments}/{progress.total_segments} "
                    f"({pct:.1f}%) - ETA: {progress.estimated_remaining:.0f}s"
                )
            
            output_path = pipeline.convert(progress_callback=progress_callback)
            logger.info(f"Created: {output_path}")
        
        except Exception as e:
            logger.error(f"Failed to convert {ebook_path.name}: {e}", exc_info=True)
            failed.append(ebook_path)
            continue
    
    if failed:
        logger.error(f"Failed to convert {len(failed)} ebook(s)")
        return 1
    
    logger.info("All conversions completed successfully")
    return 0


def run_gui(args: argparse.Namespace) -> int:
    """Run the Gradio web interface."""
    if is_port_in_use(interface_port):
        logger.error(f"Port {interface_port} is already in use")
        return 1
    
    try:
        from lib.gradio import build_interface
        
        app = build_interface(args)
        if app is None:
            logger.error("Failed to build Gradio interface")
            return 1
        
        app.queue(default_concurrency_limit=interface_concurrency_limit).launch(
            debug=debug_mode,
            show_error=debug_mode,
            favicon_path='./favicon.ico',
            server_name=interface_host,
            server_port=interface_port,
            share=args.share,
            max_file_size=max_upload_size
        )
        return 0
    
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"GUI error: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    if not check_python_version():
        return 1
    
    parser = build_parser()
    args = parser.parse_args()
    
    logger.info(f"ebook2audiobook v{prog_version}")
    
    # Handle device installation for native mode
    if args.script_mode == NATIVE:
        try:
            from lib.classes.device_installer import DeviceInstaller
            installer = DeviceInstaller()
            if installer.check_and_install_requirements():
                device_info = installer.check_device_info(args.script_mode)
                if device_info:
                    installer.install_device_packages(device_info)
        except Exception as e:
            logger.warning(f"Device setup warning: {e}")
    
    if args.headless:
        return run_headless(args)
    else:
        return run_gui(args)


if __name__ == '__main__':
    init_multiprocessing()
    sys.exit(main())
