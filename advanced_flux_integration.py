"""
Advanced Flux Integration Module
================================

Real implementation of Flux inpainting and Stable Diffusion fallback
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from PIL import Image
import cv2

try:
    import comfy.model_management as model_management
    import comfy.sample
    import comfy.samplers
    import comfy.sd
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False

try:
    from diffusers import StableDiffusionInpaintPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class AdvancedFluxEngine:
    """
    Advanced Flux integration with real inpainting capabilities
    """
    
    def __init__(self):
        if COMFY_AVAILABLE:
            self.device = model_management.get_torch_device()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.flux_model = None
        self.vae = None
        self.sd_inpaint_pipeline = None
        
    def enhance_with_inpainting(self, blended_image: np.ndarray, mask: np.ndarray,
                               positive_prompt: str, negative_prompt: str,
                               flux_model, vae, clip, strength: float = 0.3,
                               processing_mode: str = "balanced") -> Optional[np.ndarray]:
        """
        Enhanced inpainting with Flux or Stable Diffusion fallback
        """
        try:
            print("🔧 Starting advanced inpainting enhancement...")
            
            # Try Flux inpainting first
            if COMFY_AVAILABLE and flux_model is not None:
                result = self._flux_inpainting(
                    blended_image, mask, positive_prompt, negative_prompt,
                    flux_model, vae, clip, strength, processing_mode
                )
                if result is not None:
                    print("✅ Flux inpainting successful")
                    return result
                else:
                    print("⚠️ Flux inpainting failed, trying Stable Diffusion fallback")
            
            # Fallback to Stable Diffusion inpainting
            if DIFFUSERS_AVAILABLE:
                result = self._stable_diffusion_inpainting(
                    blended_image, mask, positive_prompt, negative_prompt, strength
                )
                if result is not None:
                    print("✅ Stable Diffusion inpainting successful")
                    return result
                else:
                    print("⚠️ Stable Diffusion inpainting failed")
            
            # Final fallback: advanced image processing
            print("🔄 Using advanced image processing enhancement")
            return self._image_processing_enhancement(blended_image, mask)
            
        except Exception as e:
            print(f"❌ All inpainting methods failed: {e}")
            return None
    
    def _flux_inpainting(self, image: np.ndarray, mask: np.ndarray,
                        positive_prompt: str, negative_prompt: str,
                        flux_model, vae, clip, strength: float,
                        processing_mode: str) -> Optional[np.ndarray]:
        """
        Real Flux.1 inpainting implementation
        """
        try:
            print("🚀 Executing Flux.1 inpainting...")
            
            # Convert image to tensor and encode to latent space
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Encode to latent space
            with torch.no_grad():
                latents = vae.encode(image_tensor).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            
            # Prepare mask for latent space
            mask_latent = F.interpolate(mask_tensor, size=latents.shape[-2:], mode='nearest')
            
            # Encode prompts
            positive_tokens = clip.tokenize(positive_prompt)
            negative_tokens = clip.tokenize(negative_prompt)
            
            positive_cond = clip.encode_from_tokens(positive_tokens, return_pooled=True)
            negative_cond = clip.encode_from_tokens(negative_tokens, return_pooled=True)
            
            # Configure sampling parameters based on processing mode
            if processing_mode == "fast":
                steps, cfg = 15, 6.0
            elif processing_mode == "balanced":
                steps, cfg = 25, 7.5
            else:  # quality
                steps, cfg = 35, 8.5
            
            # Prepare noise
            noise = torch.randn_like(latents)
            
            # Create inpainting latents
            masked_latents = latents * (1 - mask_latent) + noise * mask_latent
            
            # Flux sampling with inpainting
            samples = comfy.sample.sample(
                model=flux_model,
                noise=noise,
                steps=steps,
                cfg=cfg,
                sampler_name="euler_ancestral",
                scheduler="normal",
                positive=positive_cond,
                negative=negative_cond,
                latent_image=masked_latents,
                denoise=strength,
                disable_noise=False,
                start_step=0,
                last_step=steps,
                force_full_denoise=True
            )
            
            # Decode samples back to image space
            with torch.no_grad():
                decoded = vae.decode(samples["samples"]).sample
                decoded = torch.clamp(decoded, 0, 1)
            
            # Convert back to numpy
            result = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            print(f"✅ Flux inpainting completed: {result.shape}")
            return result
            
        except Exception as e:
            print(f"❌ Flux inpainting failed: {e}")
            return None
    
    def _stable_diffusion_inpainting(self, image: np.ndarray, mask: np.ndarray,
                                   positive_prompt: str, negative_prompt: str,
                                   strength: float) -> Optional[np.ndarray]:
        """
        Stable Diffusion inpainting fallback
        """
        try:
            print("🎨 Using Stable Diffusion inpainting...")
            
            # Load SD inpainting pipeline if not cached
            if self.sd_inpaint_pipeline is None:
                print("📦 Loading Stable Diffusion inpainting model...")
                self.sd_inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(self.device)
                
                # Optimize for memory
                if self.device.type == "cuda":
                    self.sd_inpaint_pipeline.enable_memory_efficient_attention()
                    self.sd_inpaint_pipeline.enable_model_cpu_offload()
            
            # Prepare inputs
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
            
            # Resize if needed (SD works best with 512x512)
            original_size = pil_image.size
            if max(original_size) > 768:
                # Resize maintaining aspect ratio
                scale = 768 / max(original_size)
                new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                pil_image = pil_image.resize(new_size, Image.LANCZOS)
                pil_mask = pil_mask.resize(new_size, Image.NEAREST)
            
            # Generate
            with torch.no_grad():
                result = self.sd_inpaint_pipeline(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    image=pil_image,
                    mask_image=pil_mask,
                    strength=strength,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    num_images_per_prompt=1,
                ).images[0]
            
            # Resize back to original size if needed
            if result.size != original_size:
                result = result.resize(original_size, Image.LANCZOS)
            
            # Convert to numpy
            result_np = np.array(result).astype(np.float32) / 255.0
            
            print(f"✅ SD inpainting completed: {result_np.shape}")
            return result_np
            
        except Exception as e:
            print(f"❌ Stable Diffusion inpainting failed: {e}")
            return None
    
    def _image_processing_enhancement(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Advanced image processing enhancement when AI inpainting fails
        """
        try:
            print("🔧 Applying advanced image processing enhancement...")
            
            # Create dilated mask for edge processing
            mask_uint8 = (mask * 255).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=2) / 255.0
            
            # Find edge region
            edge_mask = dilated_mask - mask
            edge_mask = np.clip(edge_mask, 0, 1)
            
            result = image.copy()
            
            # Apply edge-preserving filter in the logo region
            mask_region = mask > 0.5
            if mask_region.sum() > 0:
                # Bilateral filter for noise reduction while preserving edges
                filtered = cv2.bilateralFilter(
                    (image * 255).astype(np.uint8), 9, 75, 75
                ).astype(np.float32) / 255.0
                
                # Apply filtering only in mask region
                alpha = mask[:, :, np.newaxis] if len(mask.shape) == 2 else mask
                result = image * (1 - alpha * 0.3) + filtered * (alpha * 0.3)
            
            # Edge enhancement in transition region
            if edge_mask.sum() > 0:
                # Gaussian blur for smooth transitions
                blurred = cv2.GaussianBlur(result, (5, 5), 1.0)
                edge_alpha = edge_mask[:, :, np.newaxis] if len(edge_mask.shape) == 2 else edge_mask
                result = result * (1 - edge_alpha * 0.5) + blurred * (edge_alpha * 0.5)
            
            # Sharpening in logo area
            if mask.sum() > 0:
                # Unsharp mask
                gaussian = cv2.GaussianBlur(result, (3, 3), 1.0)
                sharpened = result + (result - gaussian) * 0.5
                
                mask_alpha = mask[:, :, np.newaxis] if len(mask.shape) == 2 else mask
                result = result * (1 - mask_alpha * 0.3) + sharpened * (mask_alpha * 0.3)
            
            result = np.clip(result, 0, 1)
            print("✅ Image processing enhancement completed")
            return result
            
        except Exception as e:
            print(f"❌ Image processing enhancement failed: {e}")
            return image
    
    def estimate_optimal_parameters(self, mask: np.ndarray, image_size: Tuple[int, int]) -> Dict:
        """
        Estimate optimal parameters based on mask and image properties
        """
        try:
            mask_area = (mask > 0.5).sum()
            total_area = mask.size
            mask_ratio = mask_area / total_area
            
            # Find mask complexity
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            complexity = 0
            if contours:
                # Calculate perimeter to area ratio
                largest_contour = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(largest_contour, True)
                area = cv2.contourArea(largest_contour)
                if area > 0:
                    complexity = perimeter * perimeter / (4 * np.pi * area)  # Isoperimetric ratio
            
            # Adjust parameters based on analysis
            if mask_ratio > 0.3:  # Large mask area
                strength = 0.4
                steps = 30
            elif complexity > 2.0:  # Complex shape
                strength = 0.3
                steps = 25
            else:  # Simple, small mask
                strength = 0.2
                steps = 20
            
            return {
                "inpainting_strength": strength,
                "sampling_steps": steps,
                "mask_ratio": mask_ratio,
                "complexity_score": complexity
            }
            
        except Exception as e:
            print(f"⚠️ Parameter estimation failed: {e}")
            return {
                "inpainting_strength": 0.3,
                "sampling_steps": 20,
                "mask_ratio": 0.0,
                "complexity_score": 0.0
            }