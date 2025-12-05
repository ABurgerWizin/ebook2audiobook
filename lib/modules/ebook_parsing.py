"""
Ebook parsing module for ebook2audiobook.
Handles EPUB, PDF, and other ebook format parsing.
"""

import os
import re
import shutil
import subprocess
import tempfile
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Any, Iterator
from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)


@dataclass
class Chapter:
    """Represents a chapter extracted from an ebook."""
    title: str
    content: str
    index: int
    source_file: Optional[str] = None
    
    def __str__(self) -> str:
        preview = self.content[:100].replace('\n', ' ')
        return f"Chapter {self.index}: {self.title} ({len(self.content)} chars)"


@dataclass
class EbookMetadata:
    """Metadata extracted from an ebook."""
    title: str = "Unknown"
    author: str = "Unknown"
    language: str = "en"
    cover_path: Optional[str] = None
    description: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None


@dataclass
class ParseResult:
    """Result of parsing an ebook."""
    chapters: List[Chapter]
    metadata: EbookMetadata
    source_format: str
    total_chars: int = 0
    
    def __post_init__(self):
        self.total_chars = sum(len(ch.content) for ch in self.chapters)
    
    def preview(self) -> str:
        """Generate a human-readable preview of the parsed ebook."""
        lines = [
            f"=== Ebook Parse Result ===",
            f"Title: {self.metadata.title}",
            f"Author: {self.metadata.author}",
            f"Language: {self.metadata.language}",
            f"Format: {self.source_format}",
            f"Chapters: {len(self.chapters)}",
            f"Total characters: {self.total_chars:,}",
            "",
            "Chapter listing:",
            "-" * 40
        ]
        
        for ch in self.chapters:
            lines.append(f"  {ch.index:3d}. {ch.title[:50]} ({len(ch.content):,} chars)")
        
        return "\n".join(lines)


class HTMLCleaner:
    """Utilities for cleaning HTML content from ebooks."""
    
    # Tags to remove entirely (including content)
    REMOVE_TAGS = {'script', 'style', 'nav', 'aside', 'footer', 'header'}
    
    # Tags to unwrap (keep content, remove tag)
    UNWRAP_TAGS = {'span', 'font', 'a', 'b', 'i', 'u', 'em', 'strong', 'small'}
    
    # Block-level tags that should create paragraph breaks
    BLOCK_TAGS = {'p', 'div', 'section', 'article', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                  'blockquote', 'pre', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th'}
    
    @classmethod
    def extract_text(cls, html: str) -> str:
        """Extract clean text from HTML content."""
        if not html:
            return ""
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted tags
        for tag in soup.find_all(cls.REMOVE_TAGS):
            tag.decompose()
        
        # Process the document
        return cls._process_element(soup)
    
    @classmethod
    def _process_element(cls, element) -> str:
        """Recursively process an element and its children."""
        if isinstance(element, NavigableString):
            text = str(element).strip()
            return text if text else ""
        
        if not hasattr(element, 'name'):
            return ""
        
        # Skip removed tags
        if element.name in cls.REMOVE_TAGS:
            return ""
        
        # Process children
        parts = []
        for child in element.children:
            child_text = cls._process_element(child)
            if child_text:
                parts.append(child_text)
        
        text = ' '.join(parts)
        
        # Add paragraph breaks for block elements
        if element.name in cls.BLOCK_TAGS:
            text = f"\n\n{text}\n\n"
        
        # Handle headers specially
        if element.name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            text = f"\n\n{text.strip()}\n\n"
        
        return text
    
    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """Normalize whitespace in extracted text."""
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Replace 3+ newlines with 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines).strip()


class EbookParser:
    """
    Main ebook parser supporting multiple formats.
    Uses Calibre's ebook-convert for format conversion.
    """
    
    SUPPORTED_FORMATS = {
        '.epub', '.mobi', '.azw', '.azw3', '.fb2', '.pdf',
        '.txt', '.rtf', '.doc', '.docx', '.html', '.odt'
    }
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for Calibre
        self._ebook_convert = shutil.which('ebook-convert')
    
    def parse(self, ebook_path: str) -> ParseResult:
        """
        Parse an ebook file and extract chapters.
        
        Args:
            ebook_path: Path to the ebook file
        
        Returns:
            ParseResult with chapters and metadata
        """
        path = Path(ebook_path)
        if not path.exists():
            raise FileNotFoundError(f"Ebook not found: {ebook_path}")
        
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}")
        
        # Convert to EPUB if needed (EPUB is our canonical format)
        if suffix != '.epub':
            epub_path = self._convert_to_epub(path)
        else:
            epub_path = path
        
        # Parse EPUB
        return self._parse_epub(epub_path, source_format=suffix)
    
    def _convert_to_epub(self, input_path: Path) -> Path:
        """Convert ebook to EPUB format using Calibre."""
        if not self._ebook_convert:
            raise RuntimeError("Calibre ebook-convert not found. Install Calibre for format conversion.")
        
        output_path = self.temp_dir / f"{input_path.stem}.epub"
        
        cmd = [self._ebook_convert, str(input_path), str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"ebook-convert failed: {result.stderr}")
        
        logger.info(f"Converted {input_path.name} to EPUB")
        return output_path
    
    def _parse_epub(self, epub_path: Path, source_format: str = '.epub') -> ParseResult:
        """Parse an EPUB file."""
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError:
            raise ImportError("ebooklib required: pip install ebooklib")
        
        book = epub.read_epub(str(epub_path))
        
        # Extract metadata
        metadata = self._extract_metadata(book)
        
        # Extract chapters
        chapters = self._extract_chapters(book)
        
        # Try to extract cover
        metadata.cover_path = self._extract_cover(book, epub_path)
        
        return ParseResult(
            chapters=chapters,
            metadata=metadata,
            source_format=source_format
        )
    
    def _extract_metadata(self, book) -> EbookMetadata:
        """Extract metadata from EPUB book."""
        from ebooklib import epub
        
        def get_meta(name: str) -> Optional[str]:
            try:
                values = book.get_metadata('DC', name)
                if values:
                    return values[0][0] if isinstance(values[0], tuple) else values[0]
            except Exception:
                pass
            return None
        
        return EbookMetadata(
            title=get_meta('title') or "Unknown",
            author=get_meta('creator') or "Unknown",
            language=get_meta('language') or "en",
            description=get_meta('description'),
            publisher=get_meta('publisher'),
            publication_date=get_meta('date')
        )
    
    def _extract_chapters(self, book) -> List[Chapter]:
        """Extract chapters from EPUB book."""
        from ebooklib import epub
        
        chapters = []
        chapter_idx = 0
        
        # Get spine items (reading order)
        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            
            content = item.get_content().decode('utf-8', errors='ignore')
            
            # Extract text from HTML
            text = HTMLCleaner.extract_text(content)
            text = HTMLCleaner.normalize_whitespace(text)
            
            if not text or len(text.strip()) < 50:
                continue
            
            # Try to extract chapter title
            title = self._extract_chapter_title(content, chapter_idx)
            
            chapters.append(Chapter(
                title=title,
                content=text,
                index=chapter_idx,
                source_file=item.get_name()
            ))
            chapter_idx += 1
        
        return chapters
    
    def _extract_chapter_title(self, html: str, default_idx: int) -> str:
        """Extract chapter title from HTML content."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try to find title in heading tags
        for tag in ('h1', 'h2', 'h3', 'title'):
            elem = soup.find(tag)
            if elem and elem.get_text().strip():
                title = elem.get_text().strip()
                # Limit length
                if len(title) > 100:
                    title = title[:97] + "..."
                return title
        
        # Default title
        return f"Chapter {default_idx + 1}"
    
    def _extract_cover(self, book, epub_path: Path) -> Optional[str]:
        """Extract cover image from EPUB."""
        from ebooklib import epub
        
        try:
            # Try to get cover item
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_COVER:
                    cover_path = self.temp_dir / f"{epub_path.stem}_cover.jpg"
                    cover_path.write_bytes(item.get_content())
                    return str(cover_path)
            
            # Try to find cover by name
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_IMAGE:
                    name = item.get_name().lower()
                    if 'cover' in name:
                        ext = Path(name).suffix or '.jpg'
                        cover_path = self.temp_dir / f"{epub_path.stem}_cover{ext}"
                        cover_path.write_bytes(item.get_content())
                        return str(cover_path)
        except Exception as e:
            logger.warning(f"Could not extract cover: {e}")
        
        return None
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


class ChapterExtractor:
    """
    Advanced chapter extraction with structure detection.
    Handles table of contents, nested chapters, and content classification.
    """
    
    # Patterns for detecting chapter headings
    CHAPTER_PATTERNS = [
        re.compile(r'^chapter\s+(\d+|[ivxlcdm]+)', re.IGNORECASE),
        re.compile(r'^part\s+(\d+|[ivxlcdm]+)', re.IGNORECASE),
        re.compile(r'^section\s+(\d+)', re.IGNORECASE),
        re.compile(r'^(\d+)\.\s+\w+'),  # "1. Introduction"
    ]
    
    def __init__(self, parser: Optional[EbookParser] = None):
        self.parser = parser or EbookParser()
    
    def extract(self, ebook_path: str) -> List[Chapter]:
        """Extract chapters from an ebook."""
        result = self.parser.parse(ebook_path)
        return result.chapters
    
    def merge_short_chapters(
        self,
        chapters: List[Chapter],
        min_chars: int = 500
    ) -> List[Chapter]:
        """Merge very short chapters with the following chapter."""
        if len(chapters) <= 1:
            return chapters
        
        merged = []
        buffer = None
        
        for chapter in chapters:
            if buffer is not None:
                # Merge with previous short chapter
                chapter = Chapter(
                    title=buffer.title,
                    content=buffer.content + "\n\n" + chapter.content,
                    index=buffer.index,
                    source_file=buffer.source_file
                )
                buffer = None
            
            if len(chapter.content) < min_chars:
                buffer = chapter
            else:
                merged.append(chapter)
        
        # Handle trailing buffer
        if buffer is not None:
            if merged:
                # Append to last chapter
                last = merged[-1]
                merged[-1] = Chapter(
                    title=last.title,
                    content=last.content + "\n\n" + buffer.content,
                    index=last.index,
                    source_file=last.source_file
                )
            else:
                merged.append(buffer)
        
        # Re-index
        for i, ch in enumerate(merged):
            ch.index = i
        
        return merged


