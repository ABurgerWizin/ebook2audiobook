"""
Main conversion pipeline for ebook2audiobook.
Clean implementation using the new modular components.
"""

import os
import time
import json
import logging
import uuid
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable
from collections import deque

from lib.conf import (
    tmp_dir, voices_dir, chatterbox_model_path, chatterbox_defaults,
    temp_audio_format, INFERENCE_MODE, INFERENCE_API_URL, TTS_SML
)
from lib.modules.text_processing import SmartSegmenter, BatchConfig, SegmentationResult, TextSegment
from lib.modules.ebook_parsing import EbookParser, ParseResult, Chapter
from lib.modules.audio_utils import AudioPipeline, ChapterStitcher, AudioConfig
from lib.inference import get_engine, TTSConfig

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for ebook to audiobook conversion."""
    # Input/Output
    ebook_path: str = ""
    output_dir: str = ""
    output_format: str = "m4b"
    
    # Voice
    voice_path: Optional[str] = None
    
    # TTS settings
    inference_mode: str = "local"  # 'local' or 'client'
    server_url: str = "http://localhost:8000"
    model_path: str = ""
    model_type: str = "english"
    device: str = "cuda"
    max_vram: int = 11
    
    # Generation parameters
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    language_id: str = "en"
    
    # Processing
    max_tokens_per_batch: int = 48
    temp_format: str = "flac"
    cleanup_temp: bool = True
    
    # Preview mode
    preview_only: bool = False
    preview_segments: int = 16
    
    def __post_init__(self):
        if not self.model_path:
            self.model_path = chatterbox_model_path
        if not self.output_dir:
            self.output_dir = os.path.join(tmp_dir, "audiobooks")


@dataclass
class ConversionProgress:
    """Tracks conversion progress."""
    total_chapters: int = 0
    completed_chapters: int = 0
    total_segments: int = 0
    completed_segments: int = 0
    current_chapter: str = ""
    current_segment: str = ""
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0
    
    @property
    def chapter_progress(self) -> float:
        if self.total_chapters == 0:
            return 0.0
        return self.completed_chapters / self.total_chapters
    
    @property
    def segment_progress(self) -> float:
        if self.total_segments == 0:
            return 0.0
        return self.completed_segments / self.total_segments
    
    def to_dict(self) -> dict:
        return {
            "total_chapters": self.total_chapters,
            "completed_chapters": self.completed_chapters,
            "total_segments": self.total_segments,
            "completed_segments": self.completed_segments,
            "current_chapter": self.current_chapter,
            "chapter_progress": f"{self.chapter_progress:.1%}",
            "segment_progress": f"{self.segment_progress:.1%}",
            "elapsed_time": f"{self.elapsed_time:.1f}s",
            "estimated_remaining": f"{self.estimated_remaining:.1f}s"
        }


class ConversionPipeline:
    """
    Main pipeline for converting ebooks to audiobooks.
    
    Usage:
        config = ConversionConfig(
            ebook_path="/path/to/book.epub",
            output_dir="/path/to/output",
            voice_path="/path/to/voice.wav"
        )
        
        pipeline = ConversionPipeline(config)
        
        # Preview segmentation first
        preview = pipeline.preview()
        print(preview)
        
        # Run conversion
        result = pipeline.convert()
    """
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.progress = ConversionProgress()
        self._progress_callback: Optional[Callable[[ConversionProgress], None]] = None
        self._recent_processing_times = deque(maxlen=16)
        
        # Initialize components
        self._parser: Optional[EbookParser] = None
        self._segmenter: Optional[SmartSegmenter] = None
        self._engine = None
        self._audio_pipeline: Optional[AudioPipeline] = None
        
    def _get_session_id(self) -> str:
        """Generate a short unique session ID."""
        return str(uuid.uuid4())[:8]

    def _find_existing_job(self, safe_name: str) -> Optional[Path]:
        """Find an existing incomplete job for this ebook and model."""
        base_dir = Path(tmp_dir) / "audio_work"
        if not base_dir.exists():
            return None
            
        # Look for directories starting with safe_name
        # safe_name is sanitized, so glob should be safe
        candidates = list(base_dir.glob(f"{safe_name}_*"))
        
        # Sort by modification time (newest first)
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        for candidate in candidates:
            if not candidate.is_dir():
                continue
                
            # Check for metadata file
            meta_files = list(candidate.glob("metadata_*.json"))
            if not meta_files:
                continue
                
            # Read latest metadata
            meta_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            try:
                with open(meta_files[0], 'r') as f:
                    meta = json.load(f)
                    
                # Check if it matches our current job
                # We check ebook path, voice path, model type and model path as primary keys
                
                # 1. Ebook path (Must match)
                if meta.get("ebook_path") != str(self.config.ebook_path):
                    continue
                    
                # 2. Model type (Must match)
                if meta.get("model_type") != self.config.model_type:
                    continue
                    
                # 3. Voice path (Check if present in meta, for retro-compatibility)
                current_voice = str(self.config.voice_path) if self.config.voice_path else None
                if "voice_path" in meta and meta.get("voice_path") != current_voice:
                    continue
                    
                # 4. Model path (Check if present in meta, for retro-compatibility)
                if "model_path" in meta and meta.get("model_path") != str(self.config.model_path):
                    continue
                
                return candidate
            except Exception as e:
                logger.warning(f"Failed to read metadata from {candidate}: {e}")
                continue
                
        return None

    def _manage_session(self, safe_name: str) -> str:
        """Setup session directory and handle recovery."""
        existing_job = self._find_existing_job(safe_name)
        session_path = None
        session_dir = ""
        
        if existing_job:
            logger.info(f"Found existing job: {existing_job.name}")
            session_dir = existing_job.name
            session_path = existing_job
            
            # Check metadata for compatibility
            meta_files = list(session_path.glob("metadata_*.json"))
            meta_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            try:
                if meta_files:
                    with open(meta_files[0], 'r') as f:
                        old_meta = json.load(f)
                    
                    old_max_tokens = old_meta.get("max_tokens_per_batch")
                    
                    if old_max_tokens != self.config.max_tokens_per_batch:
                        logger.info("Parameters changed (max_tokens), clearing existing chunks but keeping job...")
                        # Delete all chunks
                        for chunk in session_path.rglob("chunk_*"):
                            if chunk.suffix in ['.flac', '.wav', '.mp3']:
                                chunk.unlink()
                    else:
                        logger.info("Resuming from existing chunks...")
            except Exception as e:
                logger.warning(f"Error checking metadata: {e}")
                
        else:
            # Create new session
            session_id = self._get_session_id()
            session_dir = f"{safe_name}_{session_id}"
            session_path = Path(tmp_dir) / "audio_work" / session_dir
            session_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created new job: {session_dir}")

        # Write new metadata
        timestamp = int(time.time())
        meta = {
            "ebook_path": str(self.config.ebook_path),
            "model_type": self.config.model_type,
            "model_path": str(self.config.model_path),
            "max_tokens_per_batch": self.config.max_tokens_per_batch,
            "voice_path": self.config.voice_path,
            "language_id": self.config.language_id,
            "timestamp": timestamp,
            "config": {
                "exaggeration": self.config.exaggeration,
                "cfg_weight": self.config.cfg_weight,
                "temperature": self.config.temperature
            }
        }
        
        try:
            with open(session_path / f"metadata_{timestamp}.json", 'w') as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write metadata: {e}")
            
        return session_dir

    def _get_safe_name(self, filename: str) -> str:
        """Sanitize and shorten filename for directory usage."""
        # Get base name without extension
        name = Path(filename).stem
        # Remove non-alphanumeric characters
        safe_name = re.sub(r'[^a-zA-Z0-9]', '', name)
        # Truncate to 16 chars
        return safe_name[:16]
        
    def set_progress_callback(self, callback: Callable[[ConversionProgress], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback
    
    def _update_progress(self):
        """Notify progress callback."""
        if self._progress_callback:
            self._progress_callback(self.progress)
    
    def preview(self) -> str:
        """
        Preview how the ebook will be segmented (dry-run mode).
        
        Returns:
            Human-readable preview of segmentation
        """
        # Create session-specific temp dir
        session_id = self._get_session_id()
        safe_name = self._get_safe_name(self.config.ebook_path)
        session_dir = f"{safe_name}_{session_id}"
        
        # Parse ebook
        parser = EbookParser(temp_dir=os.path.join(tmp_dir, "preview", session_dir))
        parse_result = parser.parse(self.config.ebook_path)
        
        # Initialize segmenter
        batch_config = BatchConfig(max_tokens_per_batch=self.config.max_tokens_per_batch)
        segmenter = SmartSegmenter(batch_config)
        
        # Generate preview
        lines = [
            "=" * 60,
            "EBOOK SEGMENTATION PREVIEW",
            "=" * 60,
            "",
            f"Title: {parse_result.metadata.title}",
            f"Author: {parse_result.metadata.author}",
            f"Chapters: {len(parse_result.chapters)}",
            f"Total characters: {parse_result.total_chars:,}",
            "",
            "-" * 60,
        ]
        
        total_segments = 0
        total_tokens = 0
        
        for chapter in parse_result.chapters:
            result = segmenter.segment_text(chapter.content, chapter.index)
            total_segments += len(result.segments)
            total_tokens += result.total_tokens
            
            lines.append(f"\nChapter {chapter.index + 1}: {chapter.title}")
            lines.append(f"  Segments: {len(result.segments)}")
            lines.append(f"  Estimated tokens: {result.total_tokens}")
            lines.append(f"  Sample segments:")
            
            for seg in result.segments[:self.config.preview_segments]:
                preview = seg.text.replace('\n', ' ')
                if len(seg.text) > 400:
                    preview += "..."
                lines.append(f"    [{seg.segment_type.name:10s}] {preview}")
            
            if len(result.segments) > self.config.preview_segments:
                lines.append(f"    ... and {len(result.segments) - self.config.preview_segments} more segments")
        
        lines.extend([
            "",
            "-" * 60,
            "SUMMARY",
            "-" * 60,
            f"Total segments: {total_segments}",
            f"Total estimated tokens: {total_tokens}",
            f"Estimated audio duration: ~{total_tokens * 0.3:.0f}s ({total_tokens * 0.3 / 60:.1f} min)",
            "=" * 60,
        ])
        
        # Cleanup parser temp files
        parser.cleanup()
        
        return "\n".join(lines)
    
    def convert(self, progress_callback: Optional[Callable] = None) -> Path:
        """
        Convert ebook to audiobook.
        
        Args:
            progress_callback: Optional callback for progress updates
        
        Returns:
            Path to the generated audiobook file
        """
        if progress_callback:
            self.set_progress_callback(progress_callback)
        
        start_time = time.time()
        
        try:
            # Initialize components
            self._init_components()
            
            # Parse ebook
            logger.info(f"Parsing ebook: {self.config.ebook_path}")
            parse_result = self._parser.parse(self.config.ebook_path)
            
            self.progress.total_chapters = len(parse_result.chapters)
            
            # Pre-scan and segment all chapters to get total segments
            logger.info("Scanning chapters to calculate total progress...")
            pre_calculated_segments = {}
            total_segments = 0
            
            for chapter in parse_result.chapters:
                seg_result = self._segmenter.segment_text(chapter.content, chapter.index)
                pre_calculated_segments[chapter.index] = seg_result
                total_segments += len(seg_result.segments)
            
            self.progress.total_segments = total_segments
            logger.info(f"Total segments to process: {total_segments}")
            self._update_progress()
            
            # Process each chapter
            chapter_files = []
            chapter_titles = []
            
            for chapter in parse_result.chapters:
                logger.info(f"Processing chapter {chapter.index + 1}: {chapter.title}")
                self.progress.current_chapter = chapter.title
                
                # Check if chapter already exists (Recovery)
                if self._audio_pipeline.has_chapter(chapter.index):
                    logger.info(f"Chapter {chapter.index + 1} already exists, skipping...")
                    chapter_path = self._audio_pipeline.get_chapter_file_path(chapter.index)
                    chapter_files.append(chapter_path)
                    chapter_titles.append(chapter.title)
                    
                    self.progress.completed_chapters += 1
                    # Add segments to completed count
                    seg_result = pre_calculated_segments[chapter.index]
                    self.progress.completed_segments += len(seg_result.segments)
                    self._update_progress()
                    continue

                self._update_progress()
                
                # Use pre-calculated segments
                seg_result = pre_calculated_segments[chapter.index]
                
                # Generate audio for each segment
                for seg_idx, segment in enumerate(seg_result.segments):
                    self.progress.current_segment = segment.text[:50] + "..."
                    
                    # Skip TTS markers
                    if segment.text.strip() in TTS_SML.values():
                        continue
                    
                    # Check if chunk already exists (Recovery)
                    if self._audio_pipeline.has_chunk(chapter.index, seg_idx):
                        # Skip generation
                        self.progress.completed_segments += 1
                        
                        # Update progress display with rolling average if available
                        if self._recent_processing_times:
                            avg_time = sum(self._recent_processing_times) / len(self._recent_processing_times)
                            remaining = self.progress.total_segments - self.progress.completed_segments
                            self.progress.estimated_remaining = avg_time * remaining
                        
                        self._update_progress()
                        continue

                    # Generate audio
                    # Track processing time for this specific chunk (including save overhead)
                    # We use this for ETA calculation to separate it from total wall-clock elapsed time
                    chunk_start_time = time.time()
                    
                    result = self._engine.generate(segment.text)
                    
                    # Save chunk
                    audio_bytes = result.get_audio_bytes()
                    self._audio_pipeline.save_audio_chunk(
                        audio_data=audio_bytes,
                        chapter_idx=chapter.index,
                        chunk_idx=seg_idx,
                        sample_rate=result.sample_rate
                    )
                    
                    # Calculate chunk processing duration
                    chunk_duration = time.time() - chunk_start_time
                    
                    # Update rolling average with actual processing time
                    self._recent_processing_times.append(chunk_duration)
                    
                    self.progress.completed_segments += 1
                    
                    # Update Total Elapsed Time (Wall Clock)
                    # This tracks the total time spent in this conversion session
                    self.progress.elapsed_time = time.time() - start_time
                    
                    # Update Estimated Time to Completion (ETA)
                    # This is purely based on recent processing performance
                    if self._recent_processing_times:
                        avg_time = sum(self._recent_processing_times) / len(self._recent_processing_times)
                        remaining = self.progress.total_segments - self.progress.completed_segments
                        self.progress.estimated_remaining = avg_time * remaining
                    
                    self._update_progress()
                
                # Stitch chapter
                chapter_file = self._audio_pipeline.stitch_chapter(
                    chapter.index,
                    cleanup=self.config.cleanup_temp
                )
                chapter_files.append(chapter_file)
                chapter_titles.append(chapter.title)
                
                self.progress.completed_chapters += 1
                self._update_progress()
            
            # Create final audiobook
            output_path = Path(self.config.output_dir) / f"{parse_result.metadata.title}.{self.config.output_format}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            stitcher = ChapterStitcher(AudioConfig(output_format=self.config.output_format))
            stitcher.create_audiobook(
                chapter_files=chapter_files,
                chapter_titles=chapter_titles,
                output_path=output_path,
                metadata={
                    "title": parse_result.metadata.title,
                    "artist": parse_result.metadata.author,
                    "album": parse_result.metadata.title,
                }
            )
            
            logger.info(f"Audiobook created: {output_path}")
            return output_path
        
        finally:
            # Always release memory resources
            self._release_resources()
            
        # Only cleanup files on success
        self._cleanup_files()
        return output_path
    
    def _init_components(self):
        """Initialize all pipeline components."""
        # Generate session ID and directory name
        safe_name = self._get_safe_name(self.config.ebook_path)
        
        # Use managed session (recovery/new)
        session_dir = self._manage_session(safe_name)
        
        # Parser
        self._parser = EbookParser(
            temp_dir=os.path.join(tmp_dir, "ebook_parse", session_dir)
        )
        
        # Segmenter
        batch_config = BatchConfig(
            max_tokens_per_batch=self.config.max_tokens_per_batch
        )
        self._segmenter = SmartSegmenter(batch_config)
        
        # TTS Engine
        # Handle Turbo mode constraints
        # Turbo mode does not support exaggeration and cfg_weight parameters
        eff_exaggeration = self.config.exaggeration
        eff_cfg_weight = self.config.cfg_weight
        
        if self.config.model_type == "turbo":
            logger.info("Turbo mode selected: Disabling exaggeration and CFG weight")
            eff_exaggeration = 0.0  # Default neutral
            eff_cfg_weight = 0.0    # Default neutral
            
        tts_config = TTSConfig(
            model_path=self.config.model_path,
            model_type=self.config.model_type,
            device=self.config.device,
            max_vram=self.config.max_vram,
            exaggeration=eff_exaggeration,
            cfg_weight=eff_cfg_weight,
            temperature=self.config.temperature,
            language_id=self.config.language_id,
            reference_audio=self.config.voice_path
        )
        
        if self.config.inference_mode == 'client':
            self._engine = get_engine('client', config=tts_config, server_url=self.config.server_url)
        else:
            self._engine = get_engine('local', config=tts_config)
        
        # Set reference audio if provided
        if self.config.voice_path:
            self._engine.set_reference_audio(self.config.voice_path)
        
        # Audio pipeline
        audio_config = AudioConfig(
            temp_format=self.config.temp_format,
            output_format=self.config.output_format,
            sample_rate=self._engine.sample_rate
        )
        self._audio_pipeline = AudioPipeline(
            work_dir=os.path.join(tmp_dir, "audio_work", session_dir),
            config=audio_config
        )
    
    def _release_resources(self):
        """Release memory resources."""
        if self._engine:
            self._engine.cleanup()
        if self._parser:
            self._parser.cleanup()

    def _cleanup_files(self):
        """Cleanup temporary files."""
        if self._audio_pipeline and self.config.cleanup_temp:
            # First cleanup standard chapter files
            self._audio_pipeline.cleanup_all()
            # Then remove the session directory
            if self._audio_pipeline.work_dir.exists():
                try:
                    shutil.rmtree(self._audio_pipeline.work_dir)
                    logger.debug(f"Removed session directory: {self._audio_pipeline.work_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove session directory: {e}")

    def _cleanup(self):
        """Deprecated: Use _release_resources and _cleanup_files."""
        self._release_resources()
        self._cleanup_files()


def convert_ebook_to_audiobook(
    ebook_path: str,
    output_dir: str,
    voice_path: Optional[str] = None,
    **kwargs
) -> Path:
    """
    Convenience function to convert an ebook to audiobook.
    
    Args:
        ebook_path: Path to the ebook file
        output_dir: Output directory for the audiobook
        voice_path: Optional path to reference voice audio
        **kwargs: Additional configuration options
    
    Returns:
        Path to the generated audiobook
    """
    config = ConversionConfig(
        ebook_path=ebook_path,
        output_dir=output_dir,
        voice_path=voice_path,
        **kwargs
    )
    
    pipeline = ConversionPipeline(config)
    return pipeline.convert()


def preview_segmentation(ebook_path: str, max_tokens: int = 54, preview_segments: int = 16) -> str:
    """
    Preview how an ebook will be segmented.
    
    Args:
        ebook_path: Path to the ebook file
        max_tokens: Maximum tokens per segment
        preview_segments: Number of segments to preview per chapter
    
    Returns:
        Human-readable preview string
    """
    config = ConversionConfig(
        ebook_path=ebook_path,
        max_tokens_per_batch=max_tokens,
        preview_segments=preview_segments
    )
    
    pipeline = ConversionPipeline(config)
    return pipeline.preview()


