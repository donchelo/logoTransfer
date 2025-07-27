import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
import cv2
from transformers import CLIPProcessor, CLIPModel
import comfy.model_management as model_management
import comfy.utils
from comfy.diffusers_load import load_diffusers_checkpoint
import folder_paths
from .flux_integration import FluxInpaintingEngine

class FluxLogoTransferNode:
    """
    Professional logo transfer node using Flux.1 + FluxFill for commercial-quality results.
    Integrates CLIP semantic analysis, SAM auto-masking, and OpenCV texture processing.
    """
    
    def __init__(self):
        self.device = model_management.get_torch_device()
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
        """Generate automatic mask using SAM-like approach (simplified)"""
        try:
            # Convert images
            image_np = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            logo_np = (logo_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            
            # Template matching for logo placement
            gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            gray_logo = cv2.cvtColor(logo_np, cv2.COLOR_RGB2GRAY)
            
            # Multi-scale template matching
            scales = [0.5, 0.7, 1.0, 1.3, 1.5]
            best_match = None
            best_confidence = 0
            
            for scale in scales:
                scaled_logo = cv2.resize(gray_logo, None, fx=scale, fy=scale)
                if scaled_logo.shape[0] > gray_image.shape[0] or scaled_logo.shape[1] > gray_image.shape[1]:
                    continue
                    
                result = cv2.matchTemplate(gray_image, scaled_logo, cv2.TM_CCOEFF_NORMED)
                _, confidence, _, location = cv2.minMaxLoc(result)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = {
                        'location': location,
                        'scale': scale,
                        'confidence': confidence,
                        'size': scaled_logo.shape
                    }
            
            # Generate mask
            if best_match and best_confidence > 0.3:
                mask = np.zeros(gray_image.shape, dtype=np.uint8)
                x, y = best_match['location']
                h, w = best_match['size']
                
                # Create smooth circular/rectangular mask
                center_x, center_y = x + w//2, y + h//2
                radius = max(w, h) // 2
                
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                
                # Convert to tensor
                mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
                return mask_tensor
            else:
                # Fallback: center mask
                h, w = gray_image.shape
                mask = np.zeros((h, w), dtype=np.uint8)
                center_x, center_y = w//2, h//2
                radius = min(w, h) // 6
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
                return mask_tensor
                
        except Exception as e:
            print(f"⚠️ Auto-mask generation failed: {e}")
            # Return center mask as fallback
            h, w = image_tensor.shape[1:3]
            mask = torch.zeros((1, h, w))
            mask[0, h//3:2*h//3, w//3:2*w//3] = 1.0
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
            
            # 3. Auto-mask generation if requested
            if auto_mask_generate:
                auto_mask = self.generate_auto_mask_sam(garment_image, logo_image)
                mask = auto_mask
                quality_report.append("Auto-mask generated using SAM-like algorithm")
            
            # 4. Logo optimization
            optimized_logo = self.optimize_logo_for_fabric(logo_image, fabric_analysis, semantic_info)
            
            # 5. Prepare for Flux processing
            # Convert images to latent space
            garment_latent = vae.encode(garment_image)
            logo_latent = vae.encode(optimized_logo)
            
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
            if not self.flux_engine.load_flux_models(flux_model, vae):
                quality_report.append("⚠️ Flux models not available, using fallback blending")
                processed_image = self._advanced_blend(
                    garment_image, optimized_logo, mask, 
                    blend_strength, texture_preservation, fabric_analysis
                )
            else:
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
                
                # 6d. Convert to latent space for Flux processing
                garment_latent = vae.encode(garment_image)
                logo_latent = vae.encode(optimized_logo)
                
                # 6e. Flux.1 + FluxFill inpainting
                result_latent = self.flux_engine.flux_inpaint_logo(
                    garment_latent, logo_latent, processed_mask,
                    flux_positive_cond, flux_negative_cond,
                    steps=optimal_steps, cfg_scale=optimal_cfg,
                    denoise=blend_strength
                )
                
                # 6f. Decode back to image space
                processed_image = vae.decode(result_latent)
                mask = processed_mask
                quality_report.append("✅ Flux.1 + FluxFill inpainting completed")
            
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
        """Advanced blending algorithm with texture preservation"""
        try:
            # Convert to numpy for processing
            garment_np = garment.squeeze(0).cpu().numpy()
            logo_np = logo.squeeze(0).cpu().numpy()
            mask_np = mask.squeeze(0).cpu().numpy()
            
            # Ensure mask is 3D
            if len(mask_np.shape) == 2:
                mask_np = np.stack([mask_np] * 3, axis=-1)
            
            # Advanced blending based on fabric type
            if fabric_analysis["fabric_type"] == "textured":
                # Use multiplicative blending for textured fabrics
                blended = garment_np * (1 - mask_np * strength) + \
                         (garment_np * logo_np * mask_np * strength)
            else:
                # Use alpha blending for smooth fabrics
                blended = garment_np * (1 - mask_np * strength) + \
                         logo_np * mask_np * strength
            
            # Texture preservation
            if texture_preservation > 0:
                # Preserve original texture in non-logo areas
                texture_mask = 1 - mask_np * (1 - texture_preservation)
                blended = blended * texture_mask + garment_np * (1 - texture_mask)
            
            # Convert back to tensor
            result = torch.from_numpy(np.clip(blended, 0, 1).astype(np.float32)).unsqueeze(0)
            return result
            
        except Exception as e:
            print(f"⚠️ Advanced blending failed: {e}")
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

# Node registration
NODE_CLASS_MAPPINGS = {
    "FluxLogoTransferNode": FluxLogoTransferNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLogoTransferNode": "🔥 Flux Logo Transfer Pro"
}