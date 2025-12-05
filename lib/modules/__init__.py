"""
ebook2audiobook modules package.
Refactored from monolithic functions.py and lang.py.
"""

from .text_processing import SmartSegmenter, TextCleaner, SegmentType
from .audio_utils import AudioPipeline, ChapterStitcher
from .ebook_parsing import EbookParser, ChapterExtractor

__all__ = [
    "SmartSegmenter",
    "TextCleaner", 
    "SegmentType",
    "AudioPipeline",
    "ChapterStitcher",
    "EbookParser",
    "ChapterExtractor",
]


