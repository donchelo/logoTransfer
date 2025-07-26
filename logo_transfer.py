import torch
from .utils import tensor_to_pil, pil_to_tensor, mask_to_pil, apply_logo_with_mask

class LogoTransferNode:
    """
    Custom node para transferir logos a prendas usando máscaras
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "logo": ("IMAGE",),
                "mask": ("MASK",),
                "scale_mode": (["fit", "fill", "stretch"], {"default": "fit"}),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result_image",)
    FUNCTION = "transfer_logo"
    CATEGORY = "image/logo"
    DESCRIPTION = "Transfiere un logo a una imagen usando una máscara"
    
    def transfer_logo(self, image, logo, mask, scale_mode="fit", opacity=1.0):
        """
        Función principal que transfiere el logo a la imagen
        """
        try:
            # Convertir tensores a PIL Images
            base_pil = tensor_to_pil(image)
            logo_pil = tensor_to_pil(logo)
            mask_pil = mask_to_pil(mask)
            
            # Validar inputs
            if base_pil is None or logo_pil is None or mask_pil is None:
                raise ValueError("Error al convertir las imágenes de entrada")
            
            # Asegurar que el logo tenga canal alpha si es RGBA
            if logo_pil.mode == 'RGBA':
                logo_pil = logo_pil.convert('RGB')
            
            # Aplicar el logo con la máscara
            result_pil = apply_logo_with_mask(
                base_pil, 
                logo_pil, 
                mask_pil, 
                opacity
            )
            
            # Convertir resultado de vuelta a tensor
            result_tensor = pil_to_tensor(result_pil)
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"Error en LogoTransferNode: {str(e)}")
            # En caso de error, retornar la imagen original
            return (image,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Indica que el nodo debe recalcularse siempre"""
        return float("NaN")

# Alias para compatibilidad
NODE_CLASS_MAPPINGS = {
    "LogoTransferNode": LogoTransferNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LogoTransferNode": "Logo Transfer"
}
