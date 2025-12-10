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
    
    # Generation parameters
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    language_id: str = "en"
    
    # Processing
    max_tokens_per_batch: int = 100
    temp_format: str = "flac"
    cleanup_temp: bool = True
    
    # Preview mode
    preview_only: bool = False
    
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
        
        # Initialize components
        self._parser: Optional[EbookParser] = None
        self._segmenter: Optional[SmartSegmenter] = None
        self._engine = None
        self._audio_pipeline: Optional[AudioPipeline] = None
        
    def _get_session_id(self) -> str:
        """Generate a short unique session ID."""
        return str(uuid.uuid4())[:8]

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
            
            for seg in result.segments[:3]:
                preview = seg.text[:60].replace('\n', ' ')
                if len(seg.text) > 60:
                    preview += "..."
                lines.append(f"    [{seg.segment_type.name:10s}] {preview}")
            
            if len(result.segments) > 3:
                lines.append(f"    ... and {len(result.segments) - 3} more segments")
        
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
            self._update_progress()
            
            # Process each chapter
            chapter_files = []
            chapter_titles = []
            
            for chapter in parse_result.chapters:
                logger.info(f"Processing chapter {chapter.index + 1}: {chapter.title}")
                self.progress.current_chapter = chapter.title
                self._update_progress()
                
                # Segment chapter text
                seg_result = self._segmenter.segment_text(chapter.content, chapter.index)
                self.progress.total_segments += len(seg_result.segments)
                
                # Generate audio for each segment
                for seg_idx, segment in enumerate(seg_result.segments):
                    self.progress.current_segment = segment.text[:50] + "..."
                    
                    # Skip TTS markers
                    if segment.text.strip() in TTS_SML.values():
                        continue
                    
                    # Generate audio
                    result = self._engine.generate(segment.text)
                    
                    # Save chunk
                    audio_bytes = result.get_audio_bytes()
                    self._audio_pipeline.save_audio_chunk(
                        audio_data=audio_bytes,
                        chapter_idx=chapter.index,
                        chunk_idx=seg_idx,
                        sample_rate=result.sample_rate
                    )
                    
                    self.progress.completed_segments += 1
                    self.progress.elapsed_time = time.time() - start_time
                    
                    # Estimate remaining time
                    if self.progress.completed_segments > 0:
                        avg_time = self.progress.elapsed_time / self.progress.completed_segments
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
            self._cleanup()
    
    def _init_components(self):
        """Initialize all pipeline components."""
        # Generate session ID and directory name
        session_id = self._get_session_id()
        safe_name = self._get_safe_name(self.config.ebook_path)
        session_dir = f"{safe_name}_{session_id}"
        
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
        tts_config = TTSConfig(
            model_path=self.config.model_path,
            model_type=self.config.model_type,
            device=self.config.device,
            exaggeration=self.config.exaggeration,
            cfg_weight=self.config.cfg_weight,
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
    
    def _cleanup(self):
        """Cleanup resources."""
        if self._engine:
            self._engine.cleanup()
        if self._parser:
            self._parser.cleanup()
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


def preview_segmentation(ebook_path: str, max_tokens: int = 100) -> str:
    """
    Preview how an ebook will be segmented.
    
    Args:
        ebook_path: Path to the ebook file
        max_tokens: Maximum tokens per segment
    
    Returns:
        Human-readable preview string
    """
    config = ConversionConfig(
        ebook_path=ebook_path,
        max_tokens_per_batch=max_tokens
    )
    
    pipeline = ConversionPipeline(config)
    return pipeline.preview()


