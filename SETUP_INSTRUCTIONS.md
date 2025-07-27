# 🔥 Flux Logo Transfer - Instrucciones de Instalación

## 1. Copia archivos a RunPod

En tu RunPod, ejecuta estos comandos:

```bash
# Crear directorio en ComfyUI
mkdir -p /workspace/ComfyUI/custom_nodes/logoTransfer

# Desde tu máquina local, sube los archivos:
# - flux_logo_transfer.py
# - flux_integration.py  
# - requirements.txt
# - example_workflow.json
```

## 2. Instala dependencias adicionales

```bash
cd /workspace/ComfyUI/custom_nodes/logoTransfer
pip install scipy
```

## 3. Reinicia ComfyUI

```bash
cd /workspace/ComfyUI
python main.py --listen
```

## 4. Buscar el nodo

- En ComfyUI, presiona `Ctrl+A` (seleccionar todo) y `Delete`
- Haz clic derecho → `Add Node` → busca: **FluxLogoTransferNode**
- O busca en: `Add Node` → `FluxLogoTransfer` → `🔥 Flux Logo Transfer Pro`

## 5. Workflow básico

### Nodos necesarios:
1. **LoadImage** (x3): Imagen de prenda, logo, y máscara
2. **CheckpointLoaderSimple**: Cargar modelo Flux.1
3. **FluxLogoTransferNode**: El nodo principal
4. **SaveImage**: Guardar resultado

### Conexiones:
```
LoadImage (prenda) → garment_image
LoadImage (logo) → logo_image  
LoadImage (máscara) → mask
CheckpointLoader → flux_model, vae, clip
```

## 6. Parámetros recomendados

- **blend_strength**: 0.85 (mezcla fuerte)
- **processing_mode**: "professional" 
- **auto_mask_generate**: true (si no tienes máscara)
- **texture_analysis**: true
- **quality_enhancement**: true

## 7. Troubleshooting

### El nodo no aparece:
```bash
# Revisa logs en terminal ComfyUI
# Asegúrate que los archivos están en la ruta correcta
ls /workspace/ComfyUI/custom_nodes/logoTransfer/
```

### Error de importación:
```bash
# Instala dependencias que falten
pip install opencv-python scipy
```

### Sin modelo Flux:
- Descarga `flux1-dev.safetensors` a `/workspace/ComfyUI/models/checkpoints/`
- O usa cualquier modelo compatible con ComfyUI

## 8. Ejemplo de uso

1. Carga imagen de camiseta
2. Carga logo PNG (con transparencia preferible)
3. Opcionalmente carga máscara (área blanca = donde aplicar logo)
4. Configura parámetros
5. Ejecuta workflow

¡El resultado será una transferencia profesional del logo!