"""
ComfyUI Flux Logo Transfer - Professional Logo Integration Node

A state-of-the-art custom node for ComfyUI that enables professional-quality 
logo transfer to garments using Flux.1 + FluxFill technology.

Features:
- Flux.1 [dev] + FluxFill integration for maximum quality
- CLIP semantic analysis for intelligent garment recognition  
- SAM-like auto-masking for precise logo placement
- OpenCV texture analysis for fabric-aware processing
- Multiple quality modes: fast, professional, ultra_quality
- Commercial-grade results with texture preservation

Author: Advanced AI Logo Transfer System
Version: 1.0.0
License: MIT
"""

try:
    # Try relative import (ComfyUI)
    from .flux_logo_transfer import FluxLogoTransferNode
except ImportError:
    # Try absolute import (standalone)
    from flux_logo_transfer import FluxLogoTransferNode

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxLogoTransferNode": FluxLogoTransferNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLogoTransferNode": "Flux Logo Transfer"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Metadata
WEB_DIRECTORY = "./web"
__version__ = "1.0.0"