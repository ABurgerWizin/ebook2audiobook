"""
Gradio web interface for ebook2audiobook.
Clean implementation using the new Chatterbox-based pipeline.
"""

from __future__ import annotations

import os
import logging
import threading
import tempfile
from pathlib import Path
from typing import Optional, Generator

import gradio as gr

from lib.conf import (
    prog_version, voices_dir, ebook_formats, voice_formats,
    chatterbox_defaults, chatterbox_model_path, audiobooks_gradio_dir, tmp_dir
)
from lib.pipeline import ConversionConfig, ConversionPipeline, ConversionProgress

logger = logging.getLogger(__name__)


def get_voice_files() -> list[str]:
    """Scan voices directory for available voice files."""
    voices = []
    voices_path = Path(voices_dir)
    if voices_path.exists():
        for ext in voice_formats:
            voices.extend([str(p) for p in voices_path.rglob(f"*{ext}")])
    return sorted(voices)


def format_voice_name(path: str) -> str:
    """Format voice file path for display."""
    p = Path(path)
    # Get relative path from voices dir
    try:
        rel = p.relative_to(voices_dir)
        return str(rel.with_suffix(''))
    except ValueError:
        return p.stem


def build_interface(args) -> gr.Blocks:
    """Build the Gradio interface."""
    
    # State for tracking conversion
    conversion_state = {"running": False, "cancel": False}
    
    # Get available voices
    voice_files = get_voice_files()
    voice_choices = [(format_voice_name(v), v) for v in voice_files]
    voice_choices.insert(0, ("Default (no cloning)", ""))
    
    # Theme
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="amber",
        neutral_hue="slate",
        font=["Inter", "system-ui", "sans-serif"]
    )
    
    css = """
    .main-title { text-align: center; margin-bottom: 1rem; }
    .status-box { font-family: monospace; font-size: 0.9em; }
    .progress-text { font-size: 0.85em; color: #666; }
    """
    
    with gr.Blocks(theme=theme, css=css, title="ebook2audiobook") as app:
        gr.Markdown(f"""
        # 📚 ebook2audiobook
        **v{prog_version}** — Convert ebooks to audiobooks using Chatterbox TTS
        """, elem_classes="main-title")
        
        with gr.Tabs():
            # Main conversion tab
            with gr.Tab("Convert", id="convert"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        
                        ebook_file = gr.File(
                            label="Upload Ebook",
                            file_types=[ext for ext in ebook_formats],
                            type="filepath"
                        )
                        
                        gr.Markdown("### Voice")
                        
                        voice_dropdown = gr.Dropdown(
                            choices=voice_choices,
                            value="",
                            label="Select Voice",
                            info="Choose a pre-loaded voice or upload your own"
                        )
                        
                        voice_upload = gr.Audio(
                            label="Or Upload Voice Sample",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        
                        gr.Markdown("### Output")
                        
                        output_format = gr.Dropdown(
                            choices=["m4b", "mp3", "flac", "wav", "ogg"],
                            value="m4b",
                            label="Output Format"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Chatterbox Parameters")
                        
                        exaggeration = gr.Slider(
                            minimum=0.0, maximum=1.0,
                            value=chatterbox_defaults["exaggeration"],
                            step=0.05,
                            label="Exaggeration",
                            info="Voice expressiveness (0=neutral, 1=expressive)"
                        )
                        
                        cfg_weight = gr.Slider(
                            minimum=0.0, maximum=1.0,
                            value=chatterbox_defaults["cfg_weight"],
                            step=0.05,
                            label="CFG Weight",
                            info="Classifier-free guidance weight"
                        )
                        
                        temperature = gr.Slider(
                            minimum=0.1, maximum=1.5,
                            value=chatterbox_defaults["temperature"],
                            step=0.05,
                            label="Temperature",
                            info="Generation randomness"
                        )
                        
                        device = gr.Radio(
                            choices=["cuda", "cpu", "mps"],
                            value="cuda",
                            label="Device",
                            info="GPU recommended for faster inference"
                        )
                        
                        model_path = gr.Textbox(
                            value=chatterbox_model_path,
                            label="Model Path",
                            info="Path to Chatterbox model directory"
                        )
                
                gr.Markdown("---")
                
                with gr.Row():
                    preview_btn = gr.Button("🔍 Preview Segmentation", variant="secondary")
                    convert_btn = gr.Button("🎧 Convert to Audiobook", variant="primary", scale=2)
                    cancel_btn = gr.Button("❌ Cancel", variant="stop")
                
                progress_bar = gr.Progress()
                status_output = gr.Textbox(
                    label="Status",
                    lines=8,
                    max_lines=15,
                    elem_classes="status-box",
                    interactive=False
                )
                
                output_file = gr.File(label="Download Audiobook", visible=False)
            
            # Preview tab
            with gr.Tab("Preview", id="preview"):
                preview_output = gr.Textbox(
                    label="Segmentation Preview",
                    lines=30,
                    max_lines=50,
                    elem_classes="status-box",
                    interactive=False
                )
            
            # Settings tab
            with gr.Tab("Settings", id="settings"):
                gr.Markdown("### Advanced Settings")
                
                model_type = gr.Radio(
                    choices=["english", "multilingual"],
                    value="english",
                    label="Model Type"
                )
                
                language_id = gr.Textbox(
                    value="en",
                    label="Language Code",
                    info="For multilingual model (e.g., en, es, fr, de)"
                )
                
                max_tokens = gr.Slider(
                    minimum=50, maximum=500,
                    value=100,
                    step=10,
                    label="Max Tokens per Segment",
                    info="Larger = fewer chunks but may affect quality"
                )
        
        # Event handlers
        def run_preview(ebook_path: str, max_tok: int) -> str:
            if not ebook_path:
                return "Please upload an ebook first."
            
            try:
                from lib.pipeline import preview_segmentation
                return preview_segmentation(ebook_path, max_tokens=max_tok)
            except Exception as e:
                return f"Error: {e}"
        
        def run_conversion(
            ebook_path: str,
            voice_selected: str,
            voice_uploaded: str,
            out_format: str,
            exagg: float,
            cfg: float,
            temp: float,
            dev: str,
            model_p: str,
            model_t: str,
            lang_id: str,
            max_tok: int,
            progress: gr.Progress = gr.Progress()
        ) -> Generator:
            if not ebook_path:
                yield "Please upload an ebook.", gr.update(visible=False)
                return
            
            conversion_state["running"] = True
            conversion_state["cancel"] = False
            
            # Determine voice path
            voice_path = voice_uploaded if voice_uploaded else (voice_selected if voice_selected else None)
            
            # Setup output directory
            output_dir = Path(audiobooks_gradio_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            config = ConversionConfig(
                ebook_path=ebook_path,
                output_dir=str(output_dir),
                output_format=out_format,
                voice_path=voice_path,
                model_path=model_p,
                model_type=model_t,
                device=dev,
                exaggeration=exagg,
                cfg_weight=cfg,
                temperature=temp,
                language_id=lang_id,
                max_tokens_per_batch=max_tok
            )
            
            status_lines = [f"Starting conversion of {Path(ebook_path).name}..."]
            yield "\n".join(status_lines), gr.update(visible=False)
            
            try:
                pipeline = ConversionPipeline(config)
                
                def progress_callback(prog: ConversionProgress):
                    if conversion_state["cancel"]:
                        raise InterruptedError("Conversion cancelled by user")
                    
                    nonlocal status_lines
                    status_lines = [
                        f"Converting: {Path(ebook_path).name}",
                        f"Chapter: {prog.current_chapter}",
                        f"Progress: {prog.completed_segments}/{prog.total_segments} segments",
                        f"Elapsed: {prog.elapsed_time:.1f}s | ETA: {prog.estimated_remaining:.1f}s"
                    ]
                    progress(prog.segment_progress, desc=f"Chapter {prog.completed_chapters}/{prog.total_chapters}")
                
                output_path = pipeline.convert(progress_callback=progress_callback)
                
                conversion_state["running"] = False
                status_lines.append("")
                status_lines.append(f"✅ Conversion complete!")
                status_lines.append(f"Output: {output_path}")
                
                yield "\n".join(status_lines), gr.update(value=str(output_path), visible=True)
                
            except InterruptedError:
                conversion_state["running"] = False
                yield "Conversion cancelled.", gr.update(visible=False)
                
            except Exception as e:
                conversion_state["running"] = False
                logger.exception("Conversion failed")
                yield f"Error: {e}", gr.update(visible=False)
        
        def cancel_conversion():
            if conversion_state["running"]:
                conversion_state["cancel"] = True
                return "Cancelling..."
            return "No conversion running."
        
        # Wire up events
        preview_btn.click(
            fn=run_preview,
            inputs=[ebook_file, max_tokens],
            outputs=[preview_output]
        )
        
        convert_btn.click(
            fn=run_conversion,
            inputs=[
                ebook_file, voice_dropdown, voice_upload,
                output_format, exaggeration, cfg_weight, temperature,
                device, model_path, model_type, language_id, max_tokens
            ],
            outputs=[status_output, output_file]
        )
        
        cancel_btn.click(
            fn=cancel_conversion,
            outputs=[status_output]
        )
    
    return app
