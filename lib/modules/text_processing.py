"""
Text processing module for ebook2audiobook.
Handles intelligent text segmentation, cleaning, and normalization.
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
import logging
import tiktoken

logger = logging.getLogger(__name__)


class SegmentType(Enum):
    """Classification of text segments for appropriate TTS handling."""
    PARAGRAPH = auto()
    HEADER = auto()
    IMAGE_CAPTION = auto()
    FOOTNOTE = auto()
    QUOTE = auto()
    LIST_ITEM = auto()


@dataclass
class TextSegment:
    """A segment of text with metadata for TTS processing."""
    text: str
    segment_type: SegmentType
    estimated_tokens: int
    chapter_idx: int = 0
    paragraph_idx: int = 0
    
    def __str__(self) -> str:
        return f"[{self.segment_type.name}] {self.text[:50]}..."


@dataclass
class BatchConfig:
    """Configuration for token-budget-aware batching."""
    max_tokens_per_batch: int = 100
    min_tokens_per_batch: int = 5


@dataclass
class SegmentationResult:
    """Result of text segmentation with statistics."""
    segments: List[TextSegment]
    total_tokens: int
    num_sentences: int
    num_paragraphs: int
    
    def preview(self, max_segments: int = 20) -> str:
        """Generate a human-readable preview of segmentation."""
        lines = [
            f"=== Segmentation Preview ===",
            f"Total segments: {len(self.segments)}",
            f"Estimated tokens: {self.total_tokens}",
            f"Sentences: {self.num_sentences}",
            f"Paragraphs: {self.num_paragraphs}",
            f"",
            f"First {min(max_segments, len(self.segments))} segments:",
            "-" * 40
        ]
        for i, seg in enumerate(self.segments[:max_segments]):
            preview_text = seg.text[:80].replace('\n', ' ')
            if len(seg.text) > 80:
                preview_text += "..."
            lines.append(f"{i+1:3d}. [{seg.segment_type.name:12s}] {preview_text}")
        
        if len(self.segments) > max_segments:
            lines.append(f"... and {len(self.segments) - max_segments} more segments")
        
        return "\n".join(lines)


class TextCleaner:
    """Utilities for cleaning and normalizing text before TTS."""
    
    # Characters to remove entirely
    CHARS_REMOVE = {'\\', '|', '©', '®', '™', '*', '`', '\u00A0', '\xa0'}
    
    # Normalize quotes
    QUOTE_MAP = {
        '"': '"', '"': '"', ''': "'", ''': "'",
        '«': '"', '»': '"', '„': '"', '‟': '"'
    }
    
    @classmethod
    def clean(cls, text: str) -> str:
        """Clean text for TTS processing."""
        if not text:
            return ""
        
        # Remove XML/HTML declarations (e.g., <?xml ... ?>, <!DOCTYPE ... >)
        text = re.sub(r'<\?xml[^>]*\?>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<!DOCTYPE[^>]*>', '', text, flags=re.IGNORECASE)
        
        # Remove HTML/XML-like tags (e.g., <br>, <div>)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove unwanted characters
        for char in cls.CHARS_REMOVE:
            text = text.replace(char, '')
        
        # Remove footnotes attached to words (e.g., "Word1", "Word²")
        # Matches numbers or superscripts immediately following a non-whitespace character
        text = re.sub(r'(?<=\S)[\d\u00B2\u00B3\u00B9\u2070-\u2079]+', '', text)
        
        # Normalize quotes
        for old, new in cls.QUOTE_MAP.items():
            text = text.replace(old, new)
        
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    @classmethod
    def normalize_abbreviations(cls, text: str, abbreviations: dict) -> str:
        """Expand common abbreviations for better TTS pronunciation."""
        for abbr, expansion in abbreviations.items():
            # Match abbreviation with word boundaries
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text


class SmartSegmenter:
    """
    Context-aware text segmentation for optimal TTS generation.
    
    Key principles:
    - Sentences are atomic units (never split mid-sentence unless too long)
    - Respect document structure (headers, paragraphs, captions)
    - Batch sentences to optimize GPU utilization
    - Preserve prosodic integrity
    - Use accurate tokenization (tiktoken)
    """
    
    # Sentence boundary detection - matches period/exclamation/question
    # followed by space and capital letter (avoids splitting "Dr. Smith")
    SENTENCE_BOUNDARY = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|'  # Standard sentence end
        r'(?<=[.!?])\s*\n+',         # Sentence end at line break
        re.MULTILINE
    )
    
    # Detect headers (all caps, short lines, or markdown-style)
    HEADER_PATTERNS = [
        re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),  # Markdown headers
        re.compile(r'^[A-Z][A-Z\s]{2,50}$', re.MULTILINE),  # ALL CAPS lines
        re.compile(r'^Chapter\s+\d+', re.IGNORECASE),  # "Chapter X"
        re.compile(r'^Part\s+[IVXLCDM\d]+', re.IGNORECASE),  # "Part I/1"
    ]
    
    # Image caption patterns
    CAPTION_PATTERNS = [
        re.compile(r'^(?:Figure|Fig\.?|Image|Photo|Illustration)\s*\d*[:\.]?\s*.+', re.IGNORECASE),
        re.compile(r'^\[(?:Image|Photo|Figure).*?\]', re.IGNORECASE),
    ]
    
    # Footnote patterns
    FOOTNOTE_PATTERN = re.compile(r'^\[\d+\]|\^\d+')
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken, falling back to heuristic: {e}")
            self.tokenizer = None
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken if available, else heuristic."""
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback heuristic
        normalized = ' '.join(text.split())
        return max(1, int(len(normalized) / 4.0))
    
    def classify_segment(self, text: str) -> SegmentType:
        """Classify a text segment by its type."""
        text_stripped = text.strip()
        
        # Check for headers
        for pattern in self.HEADER_PATTERNS:
            if pattern.match(text_stripped):
                return SegmentType.HEADER
        
        # Check for image captions
        for pattern in self.CAPTION_PATTERNS:
            if pattern.match(text_stripped):
                return SegmentType.IMAGE_CAPTION
        
        # Check for footnotes
        if self.FOOTNOTE_PATTERN.match(text_stripped):
            return SegmentType.FOOTNOTE
        
        # Check for list items
        if re.match(r'^[\-\*\•]\s+|^\d+[\.\)]\s+', text_stripped):
            return SegmentType.LIST_ITEM
        
        # Check for quotes (starts and ends with quote marks)
        if (text_stripped.startswith('"') and text_stripped.endswith('"')) or \
           (text_stripped.startswith("'") and text_stripped.endswith("'")):
            return SegmentType.QUOTE
        
        return SegmentType.PARAGRAPH
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences as atomic prosodic units.
        """
        if not text.strip():
            return []
        
        sentences = self.SENTENCE_BOUNDARY.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def split_long_sentence(self, text: str, max_tokens: int) -> List[str]:
        """
        Split a long sentence into smaller chunks at natural breakpoints.
        Prioritizes splitting at commas, semicolons, colons.
        Fallback to splitting by words if no natural breaks found.
        """
        if self.estimate_tokens(text) <= max_tokens:
            return [text]
            
        # Try splitting by natural pauses first
        delimiters = [
            (r'(?<=[;:—])\s+', 1),  # Strong breaks
            (r'(?<=[,])\s+', 0.5),   # Soft breaks
        ]
        
        for pattern, _ in delimiters:
            parts = re.split(pattern, text)
            if len(parts) > 1:
                result = []
                current_chunk = []
                current_tokens = 0
                
                for part in parts:
                    part_tokens = self.estimate_tokens(part)
                    
                    # If single part is still too big, we'll need to recurse or force split
                    if part_tokens > max_tokens:
                         # Flush current chunk if any
                        if current_chunk:
                            result.append(' '.join(current_chunk))
                            current_chunk = []
                            current_tokens = 0
                        
                        # Recurse on the large part or force split
                        # Since we are already splitting by delimiters, further recursion 
                        # might not find better delimiters. We can force split at spaces.
                        result.extend(self._force_split_words(part, max_tokens))
                        continue

                    if current_tokens + part_tokens > max_tokens:
                        if current_chunk:
                            result.append(' '.join(current_chunk))
                        current_chunk = [part]
                        current_tokens = part_tokens
                    else:
                        current_chunk.append(part)
                        current_tokens += part_tokens
                
                if current_chunk:
                    result.append(' '.join(current_chunk))
                
                # Check if we successfully reduced all chunks
                if all(self.estimate_tokens(s) <= max_tokens for s in result):
                    return result
        
        # If natural splitting failed to reduce size, force split by words
        return self._force_split_words(text, max_tokens)

    def _force_split_words(self, text: str, max_tokens: int) -> List[str]:
        """Force split text by words to fit in max_tokens."""
        words = text.split()
        result = []
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self.estimate_tokens(word)
            if current_tokens + word_tokens > max_tokens:
                if current_chunk:
                    result.append(' '.join(current_chunk))
                current_chunk = [word]
                current_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_tokens += word_tokens
                
        if current_chunk:
            result.append(' '.join(current_chunk))
            
        return result

    
    def segment_text(self, text: str, chapter_idx: int = 0) -> SegmentationResult:
        """
        Segment text into optimal batches for TTS processing.
        """
        if not text:
            return SegmentationResult([], 0, 0, 0)
        
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        segments: List[TextSegment] = []
        total_tokens = 0
        num_sentences = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            seg_type = self.classify_segment(paragraph)
            
            # Headers, captions, footnotes: keep as single segments
            if seg_type in (SegmentType.HEADER, SegmentType.IMAGE_CAPTION, SegmentType.FOOTNOTE):
                tokens = self.estimate_tokens(paragraph)
                segments.append(TextSegment(
                    text=paragraph,
                    segment_type=seg_type,
                    estimated_tokens=tokens,
                    chapter_idx=chapter_idx,
                    paragraph_idx=para_idx
                ))
                total_tokens += tokens
                continue
            
            # Regular paragraphs: extract sentences and batch
            sentences = self.extract_sentences(paragraph)
            if not sentences:
                continue
            
            # Flatten sentences list (handle long sentence splitting)
            processed_sentences = []
            for sent in sentences:
                sent_tokens = self.estimate_tokens(sent)
                if sent_tokens > self.config.max_tokens_per_batch:
                    # Split long sentence
                    split_parts = self.split_long_sentence(sent, self.config.max_tokens_per_batch)
                    processed_sentences.extend(split_parts)
                else:
                    processed_sentences.append(sent)
            
            sentences = processed_sentences
            num_sentences += len(sentences)
            
            # Batch sentences by token budget
            current_batch: List[str] = []
            current_tokens = 0
            
            for sentence in sentences:
                sent_tokens = self.estimate_tokens(sentence)
                
                # Check if adding this sentence would exceed budget
                if current_tokens + sent_tokens > self.config.max_tokens_per_batch:
                    # Flush current batch
                    if current_batch:
                        batch_text = ' '.join(current_batch)
                        segments.append(TextSegment(
                            text=batch_text,
                            segment_type=seg_type,
                            estimated_tokens=current_tokens,
                            chapter_idx=chapter_idx,
                            paragraph_idx=para_idx
                        ))
                        total_tokens += current_tokens
                    
                    current_batch = [sentence]
                    current_tokens = sent_tokens
                else:
                    current_batch.append(sentence)
                    current_tokens += sent_tokens
            
            # Flush remaining batch
            if current_batch:
                batch_text = ' '.join(current_batch)
                segments.append(TextSegment(
                    text=batch_text,
                    segment_type=seg_type,
                    estimated_tokens=current_tokens,
                    chapter_idx=chapter_idx,
                    paragraph_idx=para_idx
                ))
                total_tokens += current_tokens
        
        return SegmentationResult(
            segments=segments,
            total_tokens=total_tokens,
            num_sentences=num_sentences,
            num_paragraphs=len(paragraphs)
        )
    
    def segment_for_preview(self, text: str) -> str:
        """Generate a preview of how text will be segmented (dry-run mode)."""
        result = self.segment_text(text)
        return result.preview()


# Convenience function for quick segmentation
def segment_text(text: str, max_tokens: int = 100) -> List[str]:
    """Quick helper to segment text into batches."""
    config = BatchConfig(max_tokens_per_batch=max_tokens)
    segmenter = SmartSegmenter(config)
    result = segmenter.segment_text(text)
    return [seg.text for seg in result.segments]
