# 🚀 Advanced Flux Logo Transfer Pro

## Descripción

Nodo profesional de transferencia de logos con arquitectura modular avanzada, múltiples modos de blending, inpainting real con FluxFill/Stable Diffusion, upscaling con Real-ESRGAN y métricas de calidad SSIM/LPIPS.

## 🆕 Características Avanzadas

### 📐 Arquitectura Modular Refactorizada
- **ImagePreprocessor**: Normalización y análisis de imágenes
- **AdvancedBlender**: 6 modos de blending profesionales
- **AdvancedFluxEngine**: Inpainting real con IA
- **QualityMetrics**: Evaluación objetiva de resultados
- **ModelCache**: Sistema de caché eficiente para modelos

### 🎨 Modos de Blending Avanzados
1. **Normal**: Alpha blending estándar
2. **Multiply**: Modo multiplicativo para logos oscuros
3. **Overlay**: Contraste mejorado
4. **Soft Light**: Integración suave y natural
5. **Poisson**: Blending seamless con OpenCV
6. **Multiband**: Pirámides Laplacianas para transiciones perfectas

### 🔧 Inpainting Real con IA
- **Flux.1 + FluxFill**: Inpainting de última generación
- **Stable Diffusion Fallback**: `runwayml/stable-diffusion-inpainting`
- **Image Processing Enhancement**: Procesamiento avanzado como fallback

### 📈 Post-procesado con Real-ESRGAN
- **Upscaling inteligente**: 1.0x a 4.0x
- **Mejora de nitidez**: Preservación de detalles
- **Fallback OpenCV**: Si Real-ESRGAN no está disponible

### 📊 Métricas de Calidad Objetivas
- **SSIM**: Similaridad estructural
- **LPIPS**: Distancia perceptual
- **Mask Alignment**: Precisión de colocación
- **Overall Quality Score**: Evaluación general (⭐⭐⭐⭐⭐)

## 🛠️ Instalación

### 1. Dependencias Básicas
```bash
pip install torch torchvision numpy pillow opencv-python
pip install transformers diffusers accelerate
pip install scikit-image scipy imageio
```

### 2. Dependencias Avanzadas
```bash
# Métricas de calidad
pip install lpips

# Real-ESRGAN (opcional pero recomendado)
pip install basicsr facexlib gfpgan realesrgan

# Aceleración (opcional)
pip install xformers
```

### 3. Modelos Automáticos
El nodo descarga automáticamente:
- CLIP para análisis semántico
- Stable Diffusion Inpainting (si se habilita)
- Real-ESRGAN (si se habilita upscaling)

## 📋 Parámetros del Nodo

### Inputs Requeridos
- **garment_image**: Imagen de la prenda (IMAGE)
- **logo_image**: Diseño del logo (IMAGE) 
- **mask**: Máscara manual obligatoria (MASK)
- **flux_model**: Modelo Flux.1 (MODEL)
- **vae**: VAE encoder/decoder (VAE)
- **clip**: Modelo CLIP (CLIP)

### Configuración Principal
- **blend_mode**: Modo de blending (`poisson` recomendado)
- **blend_strength**: Intensidad (0.1-1.0, default: 0.95)
- **enable_inpainting**: Activar inpainting IA (default: true)
- **enable_upscaling**: Activar Real-ESRGAN (default: false)
- **upscale_factor**: Factor de escalado (1.0-4.0x)
- **processing_mode**: Calidad (`fast`/`balanced`/`quality`)

### Parámetros Opcionales
- **positive_prompt**: Prompt positivo para inpainting
- **negative_prompt**: Prompt negativo para inpainting
- **inpainting_strength**: Intensidad del inpainting (0.1-1.0)

### Outputs
- **enhanced_image**: Imagen final procesada
- **processed_mask**: Máscara utilizada
- **quality_metrics**: Reporte de métricas de calidad
- **processing_log**: Log detallado del proceso

## 🎯 Modo de Uso Recomendado

### Para Resultados Comerciales
```json
{
  "blend_mode": "poisson",
  "blend_strength": 0.95,
  "enable_inpainting": true,
  "enable_upscaling": true,
  "upscale_factor": 2.0,
  "processing_mode": "quality",
  "inpainting_strength": 0.3
}
```

### Para Pruebas Rápidas
```json
{
  "blend_mode": "normal",
  "blend_strength": 0.8,
  "enable_inpainting": false,
  "enable_upscaling": false,
  "processing_mode": "fast"
}
```

## 📊 Interpretación de Métricas

### SSIM (Structural Similarity)
- **> 0.8**: Excelente preservación de estructura
- **0.6-0.8**: Buena calidad
- **< 0.6**: Necesita mejoras

### LPIPS (Perceptual Distance)  
- **< 0.2**: Cambios perceptualmente mínimos
- **0.2-0.4**: Cambios moderados
- **> 0.4**: Cambios significativos

### Mask Alignment
- **> 0.7**: Colocación precisa
- **0.5-0.7**: Colocación aceptable
- **< 0.5**: Problemas de alineación

### Overall Quality Score
- **⭐⭐⭐⭐⭐**: EXCELLENT - Listo para producción
- **⭐⭐⭐⭐**: GOOD - Calidad comercial
- **⭐⭐⭐**: FAIR - Necesita ajustes menores
- **⭐⭐**: NEEDS IMPROVEMENT - Revisar parámetros

## 🔄 Flujo de Procesamiento

1. **Preprocessing**: Normalización y análisis de máscara
2. **Advanced Blending**: Aplicación con modo seleccionado
3. **AI Inpainting**: Refinamiento con Flux/SD (opcional)
4. **Upscaling**: Mejora con Real-ESRGAN (opcional)
5. **Quality Assessment**: Cálculo de métricas objetivas

## 🚨 Troubleshooting

### El nodo no aparece
```bash
# Verificar instalación
ls /workspace/ComfyUI/custom_nodes/logoTransfer/
# Debe contener: advanced_logo_transfer.py, __init__.py, etc.

# Reiniciar ComfyUI completamente
```

### Error en inpainting
- Flux no disponible → Usa Stable Diffusion automáticamente
- SD no disponible → Usa procesamiento de imagen
- Revisa logs para detalles específicos

### Error en upscaling  
- Real-ESRGAN no disponible → Usa OpenCV automáticamente
- Memoria insuficiente → Reduce upscale_factor

### Métricas extrañas
- SSIM muy bajo → Revisar blend_strength
- LPIPS muy alto → Probar blend_mode diferente
- Mask alignment bajo → Verificar calidad de máscara

## 🎨 Workflows de Ejemplo

### 1. `advanced_workflow.json`
Workflow completo con todas las características activadas.

### 2. Workflow Personalizado
Crea tu propio workflow combinando:
- LoadImage (x3) para prenda, logo y máscara
- CheckpointLoaderSimple para Flux.1
- AdvancedFluxLogoTransferNode (nodo principal)
- SaveImage + PreviewImage para resultados
- ShowText (x2) para métricas y logs

## 🚀 Rendimiento y Optimización

### Memoria GPU
- **Flux Inpainting**: ~6-8GB VRAM
- **SD Inpainting**: ~4-6GB VRAM  
- **Solo Blending**: ~2GB VRAM

### Velocidad
- **Fast Mode**: ~15-30 segundos
- **Balanced Mode**: ~30-60 segundos
- **Quality Mode**: ~60-120 segundos

### Recomendaciones
- Usa `ModelCache` para evitar recargas
- `processing_mode="fast"` para iteraciones
- `processing_mode="quality"` para resultado final
- Habilita `xformers` para mayor velocidad

## 📞 Soporte

### Logs Detallados
El nodo genera logs completos en `processing_log`. Revisa para diagnosticar problemas.

### Configuración de Depuración
```python
# En advanced_logo_transfer.py, línea ~500
ENABLE_DEBUG = True  # Activa logs extra
```

¡Tu nodo profesional está listo para generar resultados de calidad comercial! 🎉