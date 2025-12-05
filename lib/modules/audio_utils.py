"""
Audio utilities for ebook2audiobook.
Handles audio format conversion, chapter stitching, and temporary file management.
"""

import os
import shutil
import subprocess
import tempfile
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, BinaryIO
import json

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    temp_format: str = 'flac'  # 'flac', 'mp3', or 'wav'
    output_format: str = 'm4b'
    sample_rate: int = 24000
    channels: int = 1  # mono
    mp3_bitrate: str = '192k'
    flac_compression: int = 5  # 0-8, higher = smaller but slower


class AudioPipeline:
    """
    Efficient audio processing pipeline with automatic cleanup.
    
    Strategy:
    1. Generate paragraph chunks as temporary files (FLAC by default)
    2. Stitch chunks into chapter files
    3. Delete temporary chunks immediately after stitching
    """
    
    def __init__(self, work_dir: str, config: Optional[AudioConfig] = None):
        self.work_dir = Path(work_dir)
        self.config = config or AudioConfig()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify ffmpeg is available
        self._ffmpeg = shutil.which('ffmpeg')
        self._ffprobe = shutil.which('ffprobe')
        if not self._ffmpeg:
            raise RuntimeError("ffmpeg not found in PATH")
    
    def _get_temp_ext(self) -> str:
        """Get file extension for temporary audio format."""
        return f'.{self.config.temp_format}'
    
    def get_chapter_dir(self, chapter_idx: int) -> Path:
        """Get/create directory for a chapter's temporary chunks."""
        chapter_dir = self.work_dir / f'chapter_{chapter_idx:03d}'
        chapter_dir.mkdir(parents=True, exist_ok=True)
        return chapter_dir
    
    def get_chunk_path(self, chapter_idx: int, chunk_idx: int) -> Path:
        """Get path for a specific chunk file."""
        chapter_dir = self.get_chapter_dir(chapter_idx)
        return chapter_dir / f'chunk_{chunk_idx:04d}{self._get_temp_ext()}'
    
    def save_audio_chunk(
        self,
        audio_data: Union[bytes, BinaryIO],
        chapter_idx: int,
        chunk_idx: int,
        sample_rate: Optional[int] = None
    ) -> Path:
        """
        Save an audio chunk to the temporary directory.
        
        Args:
            audio_data: Raw audio bytes or file-like object
            chapter_idx: Chapter index
            chunk_idx: Chunk index within chapter
            sample_rate: Sample rate (uses config default if not specified)
        
        Returns:
            Path to saved chunk file
        """
        chunk_path = self.get_chunk_path(chapter_idx, chunk_idx)
        sr = sample_rate or self.config.sample_rate
        
        if isinstance(audio_data, bytes):
            # Write raw PCM data and convert to target format
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            
            try:
                self._convert_raw_to_format(tmp_path, chunk_path, sr)
            finally:
                os.unlink(tmp_path)
        else:
            # Assume it's already in a playable format, just copy/convert
            chunk_path.write_bytes(audio_data.read())
        
        return chunk_path
    
    def _convert_raw_to_format(self, input_path: str, output_path: Path, sample_rate: int):
        """Convert raw PCM audio to the configured format."""
        cmd = [
            self._ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 's16le',  # Signed 16-bit little-endian PCM
            '-ar', str(sample_rate),
            '-ac', str(self.config.channels),
            '-i', input_path,
        ]
        
        if self.config.temp_format == 'flac':
            cmd.extend(['-compression_level', str(self.config.flac_compression)])
        elif self.config.temp_format == 'mp3':
            cmd.extend(['-b:a', self.config.mp3_bitrate])
        
        cmd.append(str(output_path))
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
    
    def stitch_chapter(
        self,
        chapter_idx: int,
        output_path: Optional[Path] = None,
        cleanup: bool = True
    ) -> Path:
        """
        Stitch all chunks of a chapter into a single file.
        
        Args:
            chapter_idx: Chapter index
            output_path: Output file path (auto-generated if not specified)
            cleanup: Delete temporary chunks after stitching
        
        Returns:
            Path to the stitched chapter file
        """
        chapter_dir = self.get_chapter_dir(chapter_idx)
        chunks = sorted(chapter_dir.glob(f'chunk_*{self._get_temp_ext()}'))
        
        if not chunks:
            raise ValueError(f"No chunks found for chapter {chapter_idx}")
        
        if output_path is None:
            output_path = self.work_dir / f'chapter_{chapter_idx:03d}.{self.config.temp_format}'
        
        # Create concat list file
        concat_list = chapter_dir / 'concat_list.txt'
        with open(concat_list, 'w') as f:
            for chunk in chunks:
                # Use relative path from concat list location
                f.write(f"file '{chunk.name}'\n")
        
        # Run ffmpeg concat
        cmd = [
            self._ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0',
            '-i', str(concat_list),
            '-c', 'copy',
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(chapter_dir))
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")
        
        logger.info(f"Stitched chapter {chapter_idx}: {len(chunks)} chunks -> {output_path}")
        
        # Cleanup temporary files
        if cleanup:
            self.cleanup_chapter(chapter_idx)
        
        return output_path
    
    def cleanup_chapter(self, chapter_idx: int):
        """Delete temporary chunk directory for a chapter."""
        chapter_dir = self.get_chapter_dir(chapter_idx)
        if chapter_dir.exists():
            shutil.rmtree(chapter_dir)
            logger.debug(f"Cleaned up temporary files for chapter {chapter_idx}")
    
    def cleanup_all(self):
        """Delete all temporary files in work directory."""
        for item in self.work_dir.iterdir():
            if item.is_dir() and item.name.startswith('chapter_'):
                shutil.rmtree(item)
        logger.info("Cleaned up all temporary audio files")
    
    def get_audio_duration(self, filepath: Union[str, Path]) -> float:
        """Get duration of an audio file in seconds."""
        cmd = [
            self._ffprobe,
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            str(filepath)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return 0.0
        
        try:
            data = json.loads(result.stdout)
            return float(data['format']['duration'])
        except (json.JSONDecodeError, KeyError, ValueError):
            return 0.0


class ChapterStitcher:
    """
    Assembles multiple chapter files into a final audiobook.
    Supports M4B with chapter markers.
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._ffmpeg = shutil.which('ffmpeg')
        if not self._ffmpeg:
            raise RuntimeError("ffmpeg not found in PATH")
    
    def create_audiobook(
        self,
        chapter_files: List[Path],
        chapter_titles: List[str],
        output_path: Path,
        metadata: Optional[dict] = None
    ) -> Path:
        """
        Create final audiobook from chapter files.
        
        Args:
            chapter_files: List of chapter audio files
            chapter_titles: List of chapter titles for markers
            output_path: Output audiobook path
            metadata: Optional metadata (title, author, etc.)
        
        Returns:
            Path to created audiobook
        """
        if len(chapter_files) != len(chapter_titles):
            raise ValueError("chapter_files and chapter_titles must have same length")
        
        # Create temporary concat list
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for chapter_file in chapter_files:
                f.write(f"file '{chapter_file.absolute()}'\n")
            concat_list = f.name
        
        # Create metadata file for chapter markers
        metadata_file = None
        if self.config.output_format in ('m4b', 'm4a', 'mp4'):
            metadata_file = self._create_chapter_metadata(chapter_files, chapter_titles)
        
        try:
            cmd = [
                self._ffmpeg, '-y', '-hide_banner', '-loglevel', 'warning',
                '-f', 'concat', '-safe', '0',
                '-i', concat_list,
            ]
            
            if metadata_file:
                cmd.extend(['-i', metadata_file])
            
            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    cmd.extend(['-metadata', f'{key}={value}'])
            
            # Output codec settings
            if self.config.output_format in ('m4b', 'm4a', 'mp4'):
                cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
                if metadata_file:
                    cmd.extend(['-map_metadata', '1'])
            elif self.config.output_format == 'mp3':
                cmd.extend(['-c:a', 'libmp3lame', '-b:a', '192k'])
            elif self.config.output_format == 'flac':
                cmd.extend(['-c:a', 'flac'])
            else:
                cmd.extend(['-c:a', 'copy'])
            
            cmd.append(str(output_path))
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg audiobook creation failed: {result.stderr}")
            
            logger.info(f"Created audiobook: {output_path}")
            return output_path
            
        finally:
            os.unlink(concat_list)
            if metadata_file:
                os.unlink(metadata_file)
    
    def _create_chapter_metadata(
        self,
        chapter_files: List[Path],
        chapter_titles: List[str]
    ) -> str:
        """Create ffmpeg metadata file with chapter markers."""
        # Calculate chapter timestamps
        timestamps = [0.0]
        for chapter_file in chapter_files[:-1]:
            duration = self._get_duration(chapter_file)
            timestamps.append(timestamps[-1] + duration)
        
        # Write metadata file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(";FFMETADATA1\n")
            
            for i, (title, start) in enumerate(zip(chapter_titles, timestamps)):
                end = timestamps[i + 1] if i + 1 < len(timestamps) else start + self._get_duration(chapter_files[i])
                
                f.write("\n[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={int(start * 1000)}\n")
                f.write(f"END={int(end * 1000)}\n")
                f.write(f"title={title}\n")
            
            return f.name
    
    def _get_duration(self, filepath: Path) -> float:
        """Get audio file duration."""
        ffprobe = shutil.which('ffprobe')
        cmd = [
            ffprobe, '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            str(filepath)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            data = json.loads(result.stdout)
            return float(data['format']['duration'])
        except (json.JSONDecodeError, KeyError, ValueError):
            return 0.0


