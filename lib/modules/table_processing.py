import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class TableProcessor:
    """
    Handles detection and linearization of tables in text and markdown.
    Converts tabular data into a TTS-friendly linear format:
    "Row X. Column Name: Cell Value. ..."
    """
    
    @classmethod
    def process_markdown(cls, text: str) -> str:
        """
        Process Markdown tables in text.
        Look for standard MD table syntax with | separator lines.
        """
        if not text:
            return ""
            
        lines = text.split('\n')
        output_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Potential start of table: line with pipes
            if '|' in line and i + 1 < len(lines):
                # Check for separator line (must contain | and - and be at least 3 chars)
                next_line = lines[i+1]
                if '|' in next_line and set(next_line.strip().replace('|', '').replace(' ', '')) == {'-'}:
                    # Found a table!
                    headers = cls._parse_row(line)
                    
                    # Skip separator line
                    current_row_idx = i + 2
                    data_rows = []
                    
                    while current_row_idx < len(lines):
                        curr_line = lines[current_row_idx]
                        if '|' not in curr_line:
                            break
                        data_rows.append(cls._parse_row(curr_line))
                        current_row_idx += 1
                        
                    # Linearize
                    table_text = cls._linearize_table(headers, data_rows)
                    output_lines.append(table_text)
                    
                    # Advance pointer
                    i = current_row_idx
                    continue
            
            output_lines.append(line)
            i += 1
            
        return '\n'.join(output_lines)

    @classmethod
    def process_text(cls, text: str) -> str:
        """
        Process ASCII/Text tables.
        Look for blocks of lines using '|' or tabs as delimiters.
        """
        if not text:
            return ""
            
        lines = text.split('\n')
        output_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # 1. Pipe-separated tables
            if line.count('|') >= 1:
                table_block = [line]
                current_row_idx = i + 1
                while current_row_idx < len(lines):
                    curr_line = lines[current_row_idx]
                    if curr_line.count('|') < 1:
                        break
                    table_block.append(curr_line)
                    current_row_idx += 1
                
                if len(table_block) >= 2:
                    headers = cls._parse_row(table_block[0])
                    data_rows = [cls._parse_row(r) for r in table_block[1:]]
                    output_lines.append(cls._linearize_table(headers, data_rows))
                    i = current_row_idx
                    continue

            # 2. Tab-separated tables (fallback)
            if line.count('\t') >= 1:
                table_block = [line]
                current_row_idx = i + 1
                while current_row_idx < len(lines):
                    curr_line = lines[current_row_idx]
                    # Must have similar tab structure to be part of same table?
                    # Or just at least 1 tab?
                    if curr_line.count('\t') < 1:
                        break
                    table_block.append(curr_line)
                    current_row_idx += 1
                
                if len(table_block) >= 2:
                    # Parse TSV
                    headers = [c.strip() for c in table_block[0].split('\t') if c.strip()]
                    data_rows = []
                    for row_str in table_block[1:]:
                        row_cells = [c.strip() for c in row_str.split('\t')]
                        # Filter out empty trailing cells if they exceed header count?
                        # Or keep them? Let's just strip.
                        data_rows.append(row_cells)
                    
                    output_lines.append(cls._linearize_table(headers, data_rows))
                    i = current_row_idx
                    continue
            
            output_lines.append(line)
            i += 1
            
        return '\n'.join(output_lines)

    @classmethod
    def _parse_row(cls, line: str) -> List[str]:
        """Parse a pipe-separated line into cells."""
        # Split by pipe, strip whitespace
        # Filter out empty strings caused by leading/trailing pipes
        cells = [c.strip() for c in line.strip('|').split('|')]
        return cells

    @classmethod
    def _linearize_table(cls, headers: List[str], rows: List[List[str]]) -> str:
        """Convert parsed table data into descriptive text."""
        if not rows:
            return ""
            
        lines = [f"\n\nTable with {len(headers)} columns.\n"]
        
        for idx, row in enumerate(rows):
            lines.append(f"Row {idx + 1}.")
            
            # Pair headers with values
            for h_idx, cell in enumerate(row):
                header = headers[h_idx] if h_idx < len(headers) else f"Column {h_idx+1}"
                # Empty cells?
                if not cell or not cell.strip():
                    cell = "empty"
                
                # TTS pause helpers
                lines.append(f"{header}: {cell}.")
            
            lines.append("\n") # Pause between rows
            
        lines.append("End of table.\n\n")
        return " ".join(lines)
