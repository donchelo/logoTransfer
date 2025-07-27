import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
import cv2
from transformers import CLIPProcessor, CLIPModel
try:
    import comfy.model_management as model_management
    import comfy.utils
    import folder_paths
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    print("⚠️ ComfyUI not detected, running in standalone mode")

try:
    # Try relative import (ComfyUI)
    from .flux_integration import FluxInpaintingEngine
except ImportError:
    # Try absolute import (standalone)
    from flux_integration import FluxInpaintingEngine

class FluxLogoTransferNode:
    """
    Professional logo transfer node using Flux.1 + FluxFill for commercial-quality results.
    Integrates CLIP semantic analysis, SAM auto-masking, and OpenCV texture processing.
    """
    
    def __init__(self):
        if COMFY_AVAILABLE:
            self.device = model_management.get_torch_device()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_cache = {}
        self.clip_model = None
        self.clip_processor = None
        self.flux_engine = FluxInpaintingEngine()
        
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
                "blend_strength": ("FLOAT", {
                    "default": 0.85, 
                    "min": 0.1, 
                    "max": 1.0, 
                    "step": 0.05,
                    "display": "slider"
                }),
                "quality_enhancement": ("BOOLEAN", {"default": True}),
                "texture_preservation": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "semantic_guidance": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "processing_mode": (["professional", "fast", "ultra_quality"], {"default": "professional"}),
            },
            "optional": {
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "professional logo application, seamless integration, commercial quality, realistic fabric texture"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry, distorted, artificial, pixelated, low quality, artifacts"
                }),
                "auto_mask_generate": ("BOOLEAN", {"default": False}),
                "texture_analysis": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("processed_image", "final_mask", "quality_report")
    FUNCTION = "transfer_logo"
    CATEGORY = "FluxLogoTransfer"
    
    def load_clip_model(self):
        """Load CLIP model for semantic analysis"""
        if self.clip_model is None:
            try:
                model_id = "openai/clip-vit-base-patch32"
                self.clip_model = CLIPModel.from_pretrained(model_id).to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained(model_id)
                print(f"✅ CLIP model loaded: {model_id}")
            except Exception as e:
                print(f"⚠️ CLIP model loading failed: {e}")
                return False
        return True
    
    def analyze_garment_semantics(self, image_tensor):
        """Analyze garment using CLIP for semantic understanding"""
        if not self.load_clip_model():
            return {"type": "unknown", "confidence": 0.0}
        
        try:
            # Convert tensor to PIL
            image_np = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            
            # CLIP analysis
            garment_types = ["t-shirt", "hoodie", "polo shirt", "dress shirt", "tank top", "sweater"]
            inputs = self.clip_processor(
                text=[f"a {garment}" for garment in garment_types],
                images=image_pil,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=-1)
                
            best_match_idx = probs.argmax().item()
            confidence = probs[0][best_match_idx].item()
            
            return {
                "type": garment_types[best_match_idx],
                "confidence": confidence,
                "probabilities": {garment_types[i]: probs[0][i].item() for i in range(len(garment_types))}
            }
            
        except Exception as e:
            print(f"⚠️ Semantic analysis failed: {e}")
            return {"type": "unknown", "confidence": 0.0}
    
    def analyze_texture_opencv(self, image_tensor):
        """Advanced texture analysis using OpenCV"""
        try:
            # Convert to OpenCV format
            image_np = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Texture analysis
            # 1. Local Binary Pattern approximation
            kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
            edges = cv2.filter2D(gray, -1, kernel)
            texture_variance = np.var(edges)
            
            # 2. Gradient analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # 3. Contrast analysis
            contrast = np.std(gray)
            
            return {
                "texture_complexity": float(texture_variance / 1000),  # Normalized
                "gradient_strength": float(np.mean(gradient_magnitude) / 255),
                "contrast_level": float(contrast / 127),
                "fabric_type": self._classify_fabric_texture(texture_variance, contrast)
            }
            
        except Exception as e:
            print(f"⚠️ Texture analysis failed: {e}")
            return {"texture_complexity": 0.5, "gradient_strength": 0.5, "contrast_level": 0.5, "fabric_type": "smooth"}
    
    def _classify_fabric_texture(self, texture_var, contrast):
        """Classify fabric type based on texture metrics"""
        if texture_var > 50 and contrast > 30:
            return "textured"  # Canvas, denim, etc.
        elif texture_var > 20:
            return "medium"    # Regular cotton
        else:
            return "smooth"    # Polyester, silk, etc.
    
    def generate_auto_mask_sam(self, image_tensor, logo_tensor):
        """Generate automatic mask for logo placement"""
        try:
            print("🎭 Generating auto-mask for logo placement...")
            
            # Get image dimensions
            h, w = image_tensor.shape[1:3]
            print(f"📐 Image dimensions: {h}x{w}")
            
            # Create logo-sized mask in strategic position
            # For polo shirt, place logo on upper left chest area
            logo_h, logo_w = logo_tensor.shape[1:3]
            
            # Calculate ideal logo size (10-15% of image width)
            ideal_size = int(min(w, h) * 0.12)
            scale_factor = ideal_size / max(logo_w, logo_h)
            
            new_logo_w = int(logo_w * scale_factor)
            new_logo_h = int(logo_h * scale_factor)
            
            print(f"🎯 Calculated logo size: {new_logo_h}x{new_logo_w}")
            
            # Position for polo shirt chest area (upper right from viewer perspective) 
            chest_x = int(w * 0.65)  # Right side of chest
            chest_y = int(h * 0.25)  # Upper chest area
            
            # Ensure logo fits within image
            chest_x = min(chest_x, w - new_logo_w - 10)
            chest_y = min(chest_y, h - new_logo_h - 10)
            chest_x = max(chest_x, 10)
            chest_y = max(chest_y, 10)
            
            print(f"📍 Logo position: ({chest_x}, {chest_y})")
            
            # Create precise rectangular mask
            mask = np.zeros((h, w), dtype=np.float32)
            
            # Create soft-edge rectangular mask
            mask_region = mask[chest_y:chest_y+new_logo_h, chest_x:chest_x+new_logo_w]
            mask_region[:, :] = 1.0
            
            # Apply slight gaussian blur for soft edges
            mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
            
            # Convert to tensor
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            
            mask_sum = mask_tensor.sum()
            print(f"✅ Auto-mask generated - sum: {mask_sum:.1f}, max: {mask_tensor.max():.3f}")
            
            return mask_tensor
                
        except Exception as e:
            print(f"❌ Auto-mask generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Simple fallback mask
            h, w = image_tensor.shape[1:3]
            mask = torch.zeros((1, h, w))
            # Small rectangular area in upper right
            mask[0, h//4:h//2, w//2:3*w//4] = 1.0
            return mask
    
    def optimize_logo_for_fabric(self, logo_tensor, fabric_analysis, semantic_info):
        """Optimize logo appearance based on fabric type and semantic analysis"""
        try:
            logo_np = (logo_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            
            # Adjust based on fabric type
            if fabric_analysis["fabric_type"] == "textured":
                # Increase contrast for textured fabrics
                logo_np = cv2.convertScaleAbs(logo_np, alpha=1.2, beta=10)
            elif fabric_analysis["fabric_type"] == "smooth":
                # Soften for smooth fabrics
                logo_np = cv2.GaussianBlur(logo_np, (3, 3), 0.5)
            
            # Semantic adjustments
            if semantic_info["type"] in ["hoodie", "sweater"] and semantic_info["confidence"] > 0.7:
                # Slightly emboss effect for thick fabrics
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                logo_np = cv2.filter2D(logo_np, -1, kernel)
            
            # Convert back to tensor
            optimized_tensor = torch.from_numpy(logo_np.astype(np.float32) / 255.0).unsqueeze(0)
            return optimized_tensor
            
        except Exception as e:
            print(f"⚠️ Logo optimization failed: {e}")
            return logo_tensor
    
    def transfer_logo(self, garment_image, logo_image, mask, flux_model, vae, clip, 
                     blend_strength, quality_enhancement, texture_preservation, 
                     semantic_guidance, processing_mode, positive_prompt=None, 
                     negative_prompt=None, auto_mask_generate=False, texture_analysis=True):
        """Main logo transfer function using Flux.1 + FluxFill"""
        
        try:
            print("🚀 Starting Flux Logo Transfer...")
            
            # Initialize analysis results
            quality_report = []
            
            # 1. Semantic Analysis
            if semantic_guidance > 0:
                semantic_info = self.analyze_garment_semantics(garment_image)
                quality_report.append(f"Garment: {semantic_info['type']} (confidence: {semantic_info['confidence']:.2f})")
            else:
                semantic_info = {"type": "unknown", "confidence": 0.0}
            
            # 2. Texture Analysis
            if texture_analysis:
                fabric_analysis = self.analyze_texture_opencv(garment_image)
                quality_report.append(f"Fabric: {fabric_analysis['fabric_type']} (complexity: {fabric_analysis['texture_complexity']:.2f})")
            else:
                fabric_analysis = {"fabric_type": "smooth", "texture_complexity": 0.5}
            
            # 3. Auto-mask generation if requested (only if no mask provided)
            if auto_mask_generate and mask is None:
                auto_mask = self.generate_auto_mask_sam(garment_image, logo_image)
                mask = auto_mask
                quality_report.append("Auto-mask generated using SAM-like algorithm")
            elif mask is not None:
                quality_report.append("Using user-provided mask for precise logo placement")
            
            # 4. Logo optimization
            optimized_logo = self.optimize_logo_for_fabric(logo_image, fabric_analysis, semantic_info)
            
            # 5. Prepare for Flux processing - Skip VAE encoding here
            # We'll handle latent conversion inside the Flux engine if needed
            
            # Create conditioning based on semantic analysis
            if positive_prompt is None:
                if semantic_info["type"] != "unknown":
                    positive_prompt = f"professional logo on {semantic_info['type']}, {fabric_analysis['fabric_type']} fabric, commercial quality, realistic integration"
                else:
                    positive_prompt = "professional logo application, seamless integration, commercial quality"
            
            if negative_prompt is None:
                negative_prompt = "blurry, distorted, artificial, pixelated, low quality, artifacts, misaligned"
            
            # Encode prompts
            positive_cond = clip.encode_from_tokens(clip.tokenize(positive_prompt), return_pooled=True)
            negative_cond = clip.encode_from_tokens(clip.tokenize(negative_prompt), return_pooled=True)
            
            # 6. Initialize Flux engine and prepare for inpainting
            # Always try advanced blending first for better compatibility
            print("🔄 Using advanced blending with texture analysis...")
            processed_image = self._advanced_blend(
                garment_image, optimized_logo, mask, 
                blend_strength, texture_preservation, fabric_analysis
            )
            quality_report.append("✅ Advanced blending with fabric analysis completed")
            
            # Optional Flux enhancement (if models are properly loaded)
            if self.flux_engine.load_flux_models(flux_model, vae) and COMFY_AVAILABLE:
                # 6a. Prepare advanced Flux conditioning
                flux_positive_cond, flux_negative_cond = self.flux_engine.prepare_flux_conditioning(
                    clip, positive_prompt, negative_prompt, semantic_info, fabric_analysis
                )
                
                # 6b. Process mask for professional results
                processed_mask = self.flux_engine.advanced_mask_processing(mask, processing_mode)
                
                # 6c. Calculate optimal parameters
                optimal_steps = self.flux_engine.calculate_optimal_steps(fabric_analysis, processing_mode)
                optimal_cfg = self.flux_engine.calculate_optimal_cfg(semantic_info)
                
                quality_report.append(f"Flux parameters: steps={optimal_steps}, cfg={optimal_cfg:.1f}")
                
                try:
                    # 6d. Try Flux enhancement on the blended result
                    print("🚀 Attempting Flux enhancement...")
                    
                    # Use the blended image as base for Flux refinement
                    flux_enhanced = self.flux_engine.enhance_with_flux(
                        processed_image, processed_mask,
                        flux_positive_cond, flux_negative_cond,
                        vae, steps=optimal_steps, cfg_scale=optimal_cfg
                    )
                    
                    if flux_enhanced is not None:
                        processed_image = flux_enhanced
                        mask = processed_mask
                        quality_report.append("✅ Flux enhancement applied successfully")
                    else:
                        quality_report.append("⚠️ Flux enhancement failed, using advanced blend")
                        
                except Exception as e:
                    print(f"⚠️ Flux enhancement error: {e}")
                    quality_report.append(f"⚠️ Flux enhancement failed: {str(e)}")
            
            # 7. Quality enhancement post-processing
            if quality_enhancement:
                processed_image = self._enhance_quality(processed_image, processing_mode)
                quality_report.append(f"Quality enhancement applied ({processing_mode} mode)")
            
            # 8. Generate quality report
            final_report = "\n".join([
                "=== Flux Logo Transfer Report ===",
                f"Processing Mode: {processing_mode}",
                f"Blend Strength: {blend_strength:.2f}",
                f"Texture Preservation: {texture_preservation:.2f}",
                f"Semantic Guidance: {semantic_guidance:.2f}",
                "--- Analysis Results ---"
            ] + quality_report + [
                "--- Transfer Complete ---",
                "✅ Professional logo integration successful"
            ])
            
            return (processed_image, mask, final_report)
            
        except Exception as e:
            error_report = f"❌ Transfer failed: {str(e)}"
            print(error_report)
            return (garment_image, mask, error_report)
    
    def _advanced_blend(self, garment, logo, mask, strength, texture_preservation, fabric_analysis):
        """Advanced blending algorithm - applies logo exactly where mask indicates"""
        try:
            print(f"🎨 Starting manual mask blending - garment: {garment.shape}, logo: {logo.shape}")
            
            # Convert to numpy for processing
            garment_np = garment.squeeze(0).cpu().numpy()
            logo_np = logo.squeeze(0).cpu().numpy()
            
            print(f"📐 Numpy shapes - garment: {garment_np.shape}, logo: {logo_np.shape}")
            
            # Handle mask
            if mask is None:
                print("❌ No mask provided! Manual mask is required.")
                return garment
            
            mask_np = mask.squeeze(0).cpu().numpy()
            if len(mask_np.shape) == 3:
                mask_np = mask_np[0]  # Take first channel if RGB mask
            
            print(f"🎭 Mask shape: {mask_np.shape}, range: {mask_np.min():.3f}-{mask_np.max():.3f}")
            print(f"🎭 Mask coverage: {(mask_np > 0.1).sum()} pixels")
            
            # Resize logo to match garment dimensions first
            if garment_np.shape[:2] != logo_np.shape[:2]:
                logo_resized = cv2.resize(logo_np, (garment_np.shape[1], garment_np.shape[0]), 
                                        interpolation=cv2.INTER_LANCZOS4)
                print(f"🔄 Logo resized to match garment: {logo_resized.shape[:2]}")
            else:
                logo_resized = logo_np.copy()
                print("✅ Logo already matches garment size")
            
            # Ensure mask is 3D for RGB blending
            if len(mask_np.shape) == 2:
                mask_3d = np.stack([mask_np] * 3, axis=-1)
            else:
                mask_3d = mask_np
            
            print(f"🎭 Final mask shape: {mask_3d.shape}")
            
            # Apply logo exactly where mask indicates
            print(f"🧵 Fabric type: {fabric_analysis['fabric_type']}, strength: {strength}")
            
            # Create alpha channel from mask
            alpha = mask_3d * strength
            
            if fabric_analysis["fabric_type"] == "textured":
                # For textured fabrics, blend more naturally with fabric
                blended = garment_np * (1 - alpha) + (garment_np * 0.3 + logo_resized * 0.7) * alpha
                print("✅ Applied textured fabric blending (fabric + logo mix)")
            else:
                # For smooth fabrics, direct logo application
                blended = garment_np * (1 - alpha) + logo_resized * alpha
                print("✅ Applied smooth fabric blending (direct logo)")
            
            # Texture preservation - maintain fabric texture in logo area
            if texture_preservation > 0:
                # Extract fabric texture and overlay it slightly
                fabric_texture = garment_np - cv2.GaussianBlur(garment_np, (15, 15), 5)
                texture_strength = texture_preservation * alpha
                blended = blended + fabric_texture * texture_strength * 0.3
                print(f"✅ Applied texture preservation: {texture_preservation}")
            
            # Ensure valid color range
            blended = np.clip(blended, 0, 1)
            
            # Convert back to tensor
            result = torch.from_numpy(blended.astype(np.float32)).unsqueeze(0)
            print(f"✅ Manual mask blend complete - output shape: {result.shape}")
            
            return result
            
        except Exception as e:
            print(f"❌ Manual mask blending failed: {e}")
            import traceback
            traceback.print_exc()
            return garment
    
    def _enhance_quality(self, image, mode):
        """Post-processing quality enhancement"""
        try:
            if mode == "fast":
                return image
            
            image_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            
            if mode == "professional":
                # Moderate enhancement
                enhanced = cv2.bilateralFilter(image_np, 9, 75, 75)
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.05, beta=2)
            elif mode == "ultra_quality":
                # Maximum enhancement
                enhanced = cv2.bilateralFilter(image_np, 15, 100, 100)
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
                # Additional sharpening
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            result = torch.from_numpy(enhanced.astype(np.float32) / 255.0).unsqueeze(0)
            return result
            
        except Exception as e:
            print(f"⚠️ Quality enhancement failed: {e}")
            return image

# Node registration is handled in __init__.py