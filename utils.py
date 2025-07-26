import torch
import numpy as np
from PIL import Image, ImageOps
import cv2

def tensor_to_pil(tensor):
    """Convierte tensor de ComfyUI a PIL Image"""
    # ComfyUI tensors vienen en formato [batch, height, width, channels]
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # Remover batch dimension
    
    # Convertir de [0,1] a [0,255] y a uint8
    array = tensor.cpu().numpy()
    array = (array * 255).astype(np.uint8)
    
    # Crear PIL Image
    if len(array.shape) == 3:
        return Image.fromarray(array, 'RGB')
    else:
        return Image.fromarray(array, 'L')

def pil_to_tensor(pil_image):
    """Convierte PIL Image a tensor de ComfyUI"""
    # Convertir a RGB si no lo es
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convertir a numpy array
    array = np.array(pil_image).astype(np.float32) / 255.0
    
    # Convertir a tensor y agregar batch dimension
    tensor = torch.from_numpy(array).unsqueeze(0)
    
    return tensor

def mask_to_pil(mask_tensor):
    """Convierte tensor de máscara a PIL Image"""
    if len(mask_tensor.shape) == 3:
        mask_tensor = mask_tensor.squeeze(0)
    
    array = mask_tensor.cpu().numpy()
    array = (array * 255).astype(np.uint8)
    
    return Image.fromarray(array, 'L')

def get_mask_bbox(mask_pil):
    """Obtiene el bounding box del área enmascarada"""
    mask_array = np.array(mask_pil)
    
    # Encontrar coordenadas donde la máscara no es cero
    coords = np.where(mask_array > 0)
    
    if len(coords[0]) == 0:
        return None  # Máscara vacía
    
    # Calcular bounding box
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    return (x_min, y_min, x_max, y_max)

def scale_logo(logo_pil, target_size, mode="fit"):
    """Escala el logo según el modo especificado"""
    target_width, target_height = target_size
    
    if mode == "fit":
        # Mantener aspect ratio, ajustar al área sin cortarlo
        logo_pil.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
        return logo_pil
    
    elif mode == "fill":
        # Mantener aspect ratio, llenar el área (puede cortar)
        logo_ratio = logo_pil.width / logo_pil.height
        target_ratio = target_width / target_height
        
        if logo_ratio > target_ratio:
            # Logo es más ancho, ajustar por altura
            new_height = target_height
            new_width = int(target_height * logo_ratio)
        else:
            # Logo es más alto, ajustar por ancho
            new_width = target_width
            new_height = int(target_width / logo_ratio)
        
        logo_resized = logo_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Centrar y recortar
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        return logo_resized.crop((left, top, right, bottom))
    
    elif mode == "stretch":
        # Estirar para llenar exactamente el área
        return logo_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)

def apply_logo_with_mask(base_img, logo_img, mask_img, opacity=1.0):
    """Aplica el logo a la imagen base usando la máscara"""
    # Asegurar que todas las imágenes tengan el mismo tamaño
    base_width, base_height = base_img.size
    mask_resized = mask_img.resize((base_width, base_height), Image.Resampling.LANCZOS)
    
    # Obtener bounding box de la máscara
    bbox = get_mask_bbox(mask_resized)
    
    if bbox is None:
        return base_img  # Máscara vacía, retornar imagen original
    
    x_min, y_min, x_max, y_max = bbox
    target_width = x_max - x_min
    target_height = y_max - y_min
    
    # Escalar logo para ajustar al área enmascarada
    logo_scaled = scale_logo(logo_img, (target_width, target_height), "fit")
    
    # Crear imagen resultado como copia de la base
    result = base_img.copy()
    
    # Calcular posición para centrar el logo en el bounding box
    logo_width, logo_height = logo_scaled.size
    paste_x = x_min + (target_width - logo_width) // 2
    paste_y = y_min + (target_height - logo_height) // 2
    
    # Crear máscara para el logo (solo el área donde debe aplicarse)
    logo_mask = Image.new('L', (base_width, base_height), 0)
    
    # Extraer la región de la máscara que corresponde al logo
    mask_region = mask_resized.crop((paste_x, paste_y, 
                                   paste_x + logo_width, 
                                   paste_y + logo_height))
    logo_mask.paste(mask_region, (paste_x, paste_y))
    
    # Aplicar opacidad
    if opacity < 1.0:
        logo_mask = logo_mask.point(lambda x: int(x * opacity))
    
    # Pegar el logo usando la máscara
    result.paste(logo_scaled, (paste_x, paste_y), logo_mask.crop((paste_x, paste_y, 
                                                                 paste_x + logo_width, 
                                                                 paste_y + logo_height)))
    
    return result
