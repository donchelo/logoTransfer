"""
Advanced Flux Logo Transfer Node - Professional Implementation
===========================================================

Modular architecture with:
- Advanced blending modes (normal, multiply, overlay, soft-light)
- Poisson and multi-band blending
- Real inpainting with FluxFill/Stable Diffusion
- Real-ESRGAN upscaling and enhancement
- Quality metrics (SSIM, LPIPS)
- Model caching system
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os
import hashlib
from typing import Dict, Tuple, Optional, List
from transformers import CLIPProcessor, CLIPModel

try:
    import comfy.model_management as model_management
    import comfy.utils
    import folder_paths
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False

try:
    from .advanced_flux_integration import AdvancedFluxEngine
except ImportError:
    from advanced_flux_integration import AdvancedFluxEngine


class ModelCache:
    """Efficient model caching system"""
    
    def __init__(self, max_cache_size: int = 3):
        self.cache = {}
        self.max_size = max_cache_size
        self.access_count = {}
        
    def get_cache_key(self, model_type: str, model_path: str) -> str:
        """Generate cache key from model info"""
        return hashlib.md5(f"{model_type}_{model_path}".encode()).hexdigest()
    
    def get_model(self, cache_key: str):
        """Retrieve model from cache"""
        if cache_key in self.cache:
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            return self.cache[cache_key]
        return None
    
    def store_model(self, cache_key: str, model):
        """Store model in cache with LRU eviction"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[cache_key] = model
        self.access_count[cache_key] = 1


class ImagePreprocessor:
    """Advanced image preprocessing module"""
    
    def __init__(self, device: torch.device):
        self.device = device
        
    def normalize_images(self, garment: torch.Tensor, logo: torch.Tensor, 
                        mask: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize and convert images to numpy arrays"""
        garment_np = garment.squeeze(0).cpu().numpy()
        logo_np = logo.squeeze(0).cpu().numpy()
        
        # Handle mask
        if mask is None:
            raise ValueError("Manual mask is required for advanced logo transfer")
            
        mask_np = mask.squeeze(0).cpu().numpy()
        if len(mask_np.shape) == 3:
            mask_np = mask_np[0]  # Take first channel
            
        return garment_np, logo_np, mask_np
    
    def resize_logo_to_garment(self, logo_np: np.ndarray, 
                              garment_shape: Tuple[int, int]) -> np.ndarray:
        """Resize logo to match garment dimensions"""
        if logo_np.shape[:2] != garment_shape[:2]:
            return cv2.resize(logo_np, (garment_shape[1], garment_shape[0]), 
                            interpolation=cv2.INTER_LANCZOS4)
        return logo_np.copy()
    
    def analyze_mask_region(self, mask_np: np.ndarray) -> Dict:
        """Analyze mask properties for optimal processing"""
        mask_binary = mask_np > 0.5
        total_pixels = mask_binary.sum()
        
        if total_pixels == 0:
            raise ValueError("No active pixels found in mask")
        
        # Find bounding box
        coords = np.where(mask_binary)
        min_y, max_y = coords[0].min(), coords[0].max()
        min_x, max_x = coords[1].min(), coords[1].max()
        
        # Calculate mask properties
        bbox_area = (max_y - min_y) * (max_x - min_x)
        fill_ratio = total_pixels / bbox_area if bbox_area > 0 else 0
        
        return {
            'total_pixels': int(total_pixels),
            'bbox': (min_x, min_y, max_x, max_y),
            'fill_ratio': float(fill_ratio),
            'is_complex': fill_ratio < 0.8  # Complex shape if not mostly filled
        }


class AdvancedBlender:
    """Advanced blending engine with multiple modes"""
    
    BLEND_MODES = ['normal', 'multiply', 'overlay', 'soft_light', 'poisson', 'multiband']
    
    def __init__(self):
        pass
    
    def blend_normal(self, base: np.ndarray, overlay: np.ndarray, 
                    mask: np.ndarray, opacity: float) -> np.ndarray:
        """Standard alpha blending"""
        alpha = mask * opacity
        if len(alpha.shape) == 2:
            alpha = np.stack([alpha] * 3, axis=-1)
        return base * (1 - alpha) + overlay * alpha
    
    def blend_multiply(self, base: np.ndarray, overlay: np.ndarray, 
                      mask: np.ndarray, opacity: float) -> np.ndarray:
        """Multiply blending mode"""
        alpha = mask * opacity
        if len(alpha.shape) == 2:
            alpha = np.stack([alpha] * 3, axis=-1)
        
        multiplied = base * overlay
        return base * (1 - alpha) + multiplied * alpha
    
    def blend_overlay(self, base: np.ndarray, overlay: np.ndarray, 
                     mask: np.ndarray, opacity: float) -> np.ndarray:
        """Overlay blending mode"""
        alpha = mask * opacity
        if len(alpha.shape) == 2:
            alpha = np.stack([alpha] * 3, axis=-1)
        
        # Overlay formula
        overlay_result = np.where(base < 0.5, 
                                 2 * base * overlay,
                                 1 - 2 * (1 - base) * (1 - overlay))
        
        return base * (1 - alpha) + overlay_result * alpha
    
    def blend_soft_light(self, base: np.ndarray, overlay: np.ndarray, 
                        mask: np.ndarray, opacity: float) -> np.ndarray:
        """Soft light blending mode"""
        alpha = mask * opacity
        if len(alpha.shape) == 2:
            alpha = np.stack([alpha] * 3, axis=-1)
        
        # Soft light formula
        soft_light = np.where(overlay < 0.5,
                             base - (1 - 2 * overlay) * base * (1 - base),
                             base + (2 * overlay - 1) * (np.sqrt(base) - base))
        
        return base * (1 - alpha) + soft_light * alpha
    
    def blend_poisson(self, base: np.ndarray, overlay: np.ndarray, 
                     mask: np.ndarray, opacity: float) -> np.ndarray:
        """Poisson blending using OpenCV"""
        try:
            # Convert to uint8 for OpenCV
            base_uint8 = (base * 255).astype(np.uint8)
            overlay_uint8 = (overlay * 255).astype(np.uint8)
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Find center of mask for Poisson blending
            coords = np.where(mask > 0.5)
            if len(coords[0]) == 0:
                return base
                
            center_y = int(np.mean(coords[0]))
            center_x = int(np.mean(coords[1]))
            center = (center_x, center_y)
            
            # Apply Poisson blending
            result = cv2.seamlessClone(overlay_uint8, base_uint8, mask_uint8, 
                                     center, cv2.NORMAL_CLONE)
            
            return result.astype(np.float32) / 255.0
            
        except Exception as e:
            print(f"⚠️ Poisson blending failed: {e}, falling back to normal blend")
            return self.blend_normal(base, overlay, mask, opacity)
    
    def blend_multiband(self, base: np.ndarray, overlay: np.ndarray, 
                       mask: np.ndarray, opacity: float, levels: int = 4) -> np.ndarray:
        """Multi-band blending using Laplacian pyramids"""
        try:
            # Create Gaussian pyramids
            base_pyramid = self._create_gaussian_pyramid(base, levels)
            overlay_pyramid = self._create_gaussian_pyramid(overlay, levels)
            mask_pyramid = self._create_gaussian_pyramid(mask, levels)
            
            # Create Laplacian pyramids
            base_laplacian = self._create_laplacian_pyramid(base_pyramid)
            overlay_laplacian = self._create_laplacian_pyramid(overlay_pyramid)
            
            # Blend each level
            blended_laplacian = []
            for i in range(levels):
                if len(mask_pyramid[i].shape) == 2:
                    mask_3d = np.stack([mask_pyramid[i]] * 3, axis=-1)
                else:
                    mask_3d = mask_pyramid[i]
                
                alpha = mask_3d * opacity
                blended_level = (base_laplacian[i] * (1 - alpha) + 
                               overlay_laplacian[i] * alpha)
                blended_laplacian.append(blended_level)
            
            # Reconstruct image from blended pyramid
            result = self._reconstruct_from_laplacian(blended_laplacian)
            return np.clip(result, 0, 1)
            
        except Exception as e:
            print(f"⚠️ Multi-band blending failed: {e}, falling back to normal blend")
            return self.blend_normal(base, overlay, mask, opacity)
    
    def _create_gaussian_pyramid(self, image: np.ndarray, levels: int) -> List[np.ndarray]:
        """Create Gaussian pyramid"""
        pyramid = [image.copy()]
        for i in range(levels - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid
    
    def _create_laplacian_pyramid(self, gaussian_pyramid: List[np.ndarray]) -> List[np.ndarray]:
        """Create Laplacian pyramid from Gaussian pyramid"""
        laplacian = []
        for i in range(len(gaussian_pyramid) - 1):
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1], 
                               dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
            laplacian.append(gaussian_pyramid[i] - expanded)
        laplacian.append(gaussian_pyramid[-1])  # Top level
        return laplacian
    
    def _reconstruct_from_laplacian(self, laplacian_pyramid: List[np.ndarray]) -> np.ndarray:
        """Reconstruct image from Laplacian pyramid"""
        image = laplacian_pyramid[-1]
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            expanded = cv2.pyrUp(image, 
                               dstsize=(laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]))
            image = expanded + laplacian_pyramid[i]
        return image
    
    def apply_blend(self, base: np.ndarray, overlay: np.ndarray, 
                   mask: np.ndarray, mode: str, opacity: float) -> np.ndarray:
        """Apply specified blending mode"""
        if mode == 'normal':
            return self.blend_normal(base, overlay, mask, opacity)
        elif mode == 'multiply':
            return self.blend_multiply(base, overlay, mask, opacity)
        elif mode == 'overlay':
            return self.blend_overlay(base, overlay, mask, opacity)
        elif mode == 'soft_light':
            return self.blend_soft_light(base, overlay, mask, opacity)
        elif mode == 'poisson':
            return self.blend_poisson(base, overlay, mask, opacity)
        elif mode == 'multiband':
            return self.blend_multiband(base, overlay, mask, opacity)
        else:
            raise ValueError(f"Unknown blend mode: {mode}")


class QualityMetrics:
    """Quality assessment metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM between two images"""
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            else:
                img1_gray, img2_gray = img1, img2
            
            return ssim(img1_gray, img2_gray, data_range=1.0)
            
        except ImportError:
            print("⚠️ scikit-image not available for SSIM calculation")
            return 0.0
        except Exception as e:
            print(f"⚠️ SSIM calculation failed: {e}")
            return 0.0
    
    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate LPIPS perceptual distance"""
        try:
            import lpips
            
            # Initialize LPIPS model (cached)
            if not hasattr(self, 'lpips_model'):
                self.lpips_model = lpips.LPIPS(net='alex')
            
            # Ensure tensors are in correct format
            if img1.dim() == 3:
                img1 = img1.unsqueeze(0)
            if img2.dim() == 3:
                img2 = img2.unsqueeze(0)
            
            # Convert from [0,1] to [-1,1]
            img1 = img1 * 2 - 1
            img2 = img2 * 2 - 1
            
            with torch.no_grad():
                distance = self.lpips_model(img1, img2)
            
            return float(distance.item())
            
        except ImportError:
            print("⚠️ LPIPS not available for perceptual distance calculation")
            return 0.0
        except Exception as e:
            print(f"⚠️ LPIPS calculation failed: {e}")
            return 0.0
    
    def calculate_mask_alignment(self, mask: np.ndarray, result: np.ndarray, 
                               original: np.ndarray) -> float:
        """Calculate how well the logo aligns with the mask"""
        try:
            mask_binary = mask > 0.5
            if mask_binary.sum() == 0:
                return 0.0
            
            # Calculate change in mask area
            diff = np.abs(result - original)
            mask_change = np.mean(diff[mask_binary])
            non_mask_change = np.mean(diff[~mask_binary]) if (~mask_binary).sum() > 0 else 0
            
            # Good alignment = high change in mask area, low change outside
            alignment_score = mask_change / (mask_change + non_mask_change + 1e-8)
            return float(np.clip(alignment_score, 0, 1))
            
        except Exception as e:
            print(f"⚠️ Mask alignment calculation failed: {e}")
            return 0.0


# Continue with the main node class in the next part...
class AdvancedFluxLogoTransferNode:
    """
    Advanced Flux Logo Transfer Node with professional features
    """
    
    def __init__(self):
        if COMFY_AVAILABLE:
            self.device = model_management.get_torch_device()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize modules
        self.model_cache = ModelCache()
        self.preprocessor = ImagePreprocessor(self.device)
        self.blender = AdvancedBlender()
        self.quality_metrics = QualityMetrics()
        self.flux_engine = AdvancedFluxEngine()
        
        # Initialize models
        self.clip_model = None
        self.clip_processor = None
        self.esrgan_model = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "garment_image": ("IMAGE",),
                "logo_image": ("IMAGE",),
                "mask": ("MASK",),
                "flux_model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "blend_mode": (AdvancedBlender.BLEND_MODES, {"default": "poisson"}),
                "blend_strength": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "enable_inpainting": ("BOOLEAN", {"default": True}),
                "enable_upscaling": ("BOOLEAN", {"default": False}),
                "upscale_factor": ("FLOAT", {
                    "default": 1.5,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "processing_mode": (["fast", "balanced", "quality"], {"default": "balanced"}),
            },
            "optional": {
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "professional logo integration, seamless blending, high quality commercial result"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry edges, artifacts, poor integration, distorted logo, low quality"
                }),
                "inpainting_strength": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_image", "processed_mask", "quality_metrics", "processing_log")
    FUNCTION = "advanced_transfer_logo"
    CATEGORY = "AdvancedLogoTransfer"
    
    def advanced_transfer_logo(self, garment_image, logo_image, mask, flux_model, vae, clip,
                             blend_mode, blend_strength, enable_inpainting, enable_upscaling,
                             upscale_factor, processing_mode, positive_prompt=None,
                             negative_prompt=None, inpainting_strength=0.3):
        """
        Advanced logo transfer with professional features
        """
        processing_log = []
        quality_report = {}
        
        try:
            processing_log.append("🚀 Starting Advanced Flux Logo Transfer...")
            
            # Step 1: Preprocessing
            processing_log.append("📐 Step 1: Image Preprocessing")
            garment_np, logo_np, mask_np = self.preprocessor.normalize_images(
                garment_image, logo_image, mask)
            
            logo_resized = self.preprocessor.resize_logo_to_garment(logo_np, garment_np.shape)
            mask_analysis = self.preprocessor.analyze_mask_region(mask_np)
            
            processing_log.append(f"   Mask analysis: {mask_analysis['total_pixels']} pixels, "
                                f"fill ratio: {mask_analysis['fill_ratio']:.3f}")
            
            # Step 2: Advanced Blending
            processing_log.append(f"🎨 Step 2: Advanced Blending (mode: {blend_mode})")
            blended_result = self.blender.apply_blend(
                garment_np, logo_resized, mask_np, blend_mode, blend_strength)
            
            processing_log.append(f"   Blending complete with {blend_mode} mode")
            
            # Step 3: Inpainting Enhancement (if enabled)
            if enable_inpainting:
                processing_log.append("🔧 Step 3: Flux Inpainting Enhancement")
                inpainted_result = self.flux_engine.enhance_with_inpainting(
                    blended_result, mask_np, positive_prompt, negative_prompt,
                    flux_model, vae, clip, inpainting_strength, processing_mode)
                
                if inpainted_result is not None:
                    blended_result = inpainted_result
                    processing_log.append("   ✅ Inpainting enhancement applied")
                else:
                    processing_log.append("   ⚠️ Inpainting failed, using blended result")
            
            # Step 4: Upscaling (if enabled)
            final_result = blended_result
            if enable_upscaling and upscale_factor > 1.0:
                processing_log.append(f"📈 Step 4: Real-ESRGAN Upscaling ({upscale_factor}x)")
                upscaled_result = self._apply_esrgan_upscaling(blended_result, upscale_factor)
                
                if upscaled_result is not None:
                    final_result = upscaled_result
                    processing_log.append(f"   ✅ Upscaled to {final_result.shape[:2]}")
                else:
                    processing_log.append("   ⚠️ Upscaling failed, using original resolution")
            
            # Step 5: Quality Assessment
            processing_log.append("📊 Step 5: Quality Assessment")
            
            # Calculate metrics
            ssim_score = self.quality_metrics.calculate_ssim(garment_np, final_result)
            alignment_score = self.quality_metrics.calculate_mask_alignment(
                mask_np, final_result, garment_np)
            
            # LPIPS calculation (requires tensor conversion)
            garment_tensor = torch.from_numpy(garment_np).permute(2, 0, 1).unsqueeze(0)
            result_tensor = torch.from_numpy(final_result).permute(2, 0, 1).unsqueeze(0)
            lpips_score = self.quality_metrics.calculate_lpips(garment_tensor, result_tensor)
            
            quality_report = {
                "ssim": float(ssim_score),
                "lpips": float(lpips_score),
                "mask_alignment": float(alignment_score),
                "blend_mode": blend_mode,
                "blend_strength": float(blend_strength),
                "processing_mode": processing_mode,
                "inpainting_applied": enable_inpainting,
                "upscaling_applied": enable_upscaling and upscale_factor > 1.0,
                "final_resolution": final_result.shape[:2]
            }
            
            processing_log.append(f"   SSIM: {ssim_score:.4f}")
            processing_log.append(f"   LPIPS: {lpips_score:.4f}")
            processing_log.append(f"   Mask Alignment: {alignment_score:.4f}")
            
            # Convert back to tensor format
            result_tensor = torch.from_numpy(final_result.astype(np.float32)).unsqueeze(0)
            processed_mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)
            
            # Generate reports
            processing_log.append("✅ Advanced logo transfer completed successfully!")
            log_text = "\n".join(processing_log)
            metrics_text = self._format_quality_report(quality_report)
            
            return (result_tensor, processed_mask, metrics_text, log_text)
            
        except Exception as e:
            error_msg = f"❌ Advanced transfer failed: {str(e)}"
            processing_log.append(error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            return (garment_image, mask, f"Error: {str(e)}", "\n".join(processing_log))
    
    def _apply_esrgan_upscaling(self, image: np.ndarray, scale_factor: float) -> Optional[np.ndarray]:
        """Apply Real-ESRGAN upscaling"""
        try:
            # Try to load Real-ESRGAN model
            if self.esrgan_model is None:
                try:
                    from realesrgan import RealESRGANer
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                  num_block=23, num_grow_ch=32, scale=int(scale_factor))
                    
                    self.esrgan_model = RealESRGANer(
                        scale=int(scale_factor),
                        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                        model=model,
                        tile=512,
                        tile_pad=10,
                        pre_pad=0,
                        half=True
                    )
                except ImportError:
                    print("⚠️ Real-ESRGAN not available, using OpenCV upscaling")
                    return self._opencv_upscale(image, scale_factor)
            
            # Apply Real-ESRGAN
            image_uint8 = (image * 255).astype(np.uint8)
            upscaled, _ = self.esrgan_model.enhance(image_uint8, outscale=scale_factor)
            return upscaled.astype(np.float32) / 255.0
            
        except Exception as e:
            print(f"⚠️ Real-ESRGAN upscaling failed: {e}, using OpenCV fallback")
            return self._opencv_upscale(image, scale_factor)
    
    def _opencv_upscale(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """Fallback OpenCV upscaling"""
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    def _format_quality_report(self, metrics: Dict) -> str:
        """Format quality metrics into readable report"""
        report_lines = [
            "=== ADVANCED LOGO TRANSFER QUALITY REPORT ===",
            f"Processing Mode: {metrics.get('processing_mode', 'unknown')}",
            f"Blend Mode: {metrics.get('blend_mode', 'unknown')}",
            f"Blend Strength: {metrics.get('blend_strength', 0):.2%}",
            "",
            "--- Quality Metrics ---",
            f"SSIM (Structural Similarity): {metrics.get('ssim', 0):.4f}",
            f"LPIPS (Perceptual Distance): {metrics.get('lpips', 0):.4f}",
            f"Mask Alignment Score: {metrics.get('mask_alignment', 0):.4f}",
            "",
            "--- Processing Applied ---",
            f"Inpainting Enhancement: {'✅' if metrics.get('inpainting_applied') else '❌'}",
            f"Real-ESRGAN Upscaling: {'✅' if metrics.get('upscaling_applied') else '❌'}",
            f"Final Resolution: {metrics.get('final_resolution', 'unknown')}",
            "",
            "--- Quality Assessment ---"
        ]
        
        # Quality scoring
        ssim = metrics.get('ssim', 0)
        lpips = metrics.get('lpips', 1)
        alignment = metrics.get('mask_alignment', 0)
        
        if ssim > 0.8 and lpips < 0.2 and alignment > 0.7:
            report_lines.append("Overall Quality: EXCELLENT ⭐⭐⭐⭐⭐")
        elif ssim > 0.6 and lpips < 0.4 and alignment > 0.5:
            report_lines.append("Overall Quality: GOOD ⭐⭐⭐⭐")
        elif ssim > 0.4 and lpips < 0.6 and alignment > 0.3:
            report_lines.append("Overall Quality: FAIR ⭐⭐⭐")
        else:
            report_lines.append("Overall Quality: NEEDS IMPROVEMENT ⭐⭐")
        
        report_lines.append("=" * 45)
        
        return "\n".join(report_lines)


# Node registration
NODE_CLASS_MAPPINGS = {
    "AdvancedFluxLogoTransferNode": AdvancedFluxLogoTransferNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedFluxLogoTransferNode": "Advanced Flux Logo Transfer Pro"
}