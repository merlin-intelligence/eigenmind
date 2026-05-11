"""Chunking strategies — Langchain-based and ChunkNorris-based."""
from __future__ import annotations

from chunknorris.chunkers import MarkdownChunker
from chunknorris.parsers import PdfParser
from chunknorris.pipelines import BasePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

from eigenmind.config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    ocr_available,
)


def langchain_chunker(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """Recursive char-level splitter from langchain. Cheap and language-agnostic."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHUNK_SEPARATORS,
    )


def chunknorris_pipeline():
    """High-quality PDF → markdown → chunks pipeline. Heavier but better structure."""
    return BasePipeline(
        parser=PdfParser(use_ocr="auto" if ocr_available() else "never"),
        chunker=MarkdownChunker(),
    )
