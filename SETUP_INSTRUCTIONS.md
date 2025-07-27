# Flux Logo Transfer - Instrucciones de Instalación

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

- En ComfyUI, haz clic derecho → `Add Node` 
- Busca: **Flux Logo Transfer** (nombre simple, sin emojis)
- También puedes buscar: **FluxLogoTransferNode**
- Categoría: FluxLogoTransfer

## 5. Workflows de ejemplo

Tienes 2 workflows listos para usar:

### A) **flux_logo_workflow_clean.json** (Recomendado)
- **Auto-máscara**: Genera máscara automáticamente 
- **Simple**: Solo necesitas 2 imágenes (prenda + logo)
- **Rápido**: Ideal para pruebas iniciales

### B) **flux_logo_workflow_with_mask.json** (Avanzado)  
- **Máscara manual**: Puedes proporcionar tu propia máscara
- **Control total**: Mayor precisión en la colocación
- **Análisis completo**: Muestra máscara usada y reporte detallado

### Cargar workflow:
```
1. Arrastra el archivo .json a ComfyUI
2. O usa "Load" en la interfaz
3. Conecta tus imágenes
4. ¡Ejecuta!
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