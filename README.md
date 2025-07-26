# Logo Transfer Node

Custom node para ComfyUI que permite transferir logos a imágenes de prendas usando máscaras.

## Características

- ✅ Transferencia precisa de logos usando máscaras
- ✅ Múltiples modos de escalado (fit, fill, stretch)
- ✅ Control de opacidad
- ✅ Preservación de calidad de imagen
- ✅ Compatible con diferentes formatos de imagen

## Instalación

1. Copia la carpeta `logo_transfer_node` en tu directorio `ComfyUI/custom_nodes/`
2. Reinicia ComfyUI
3. El nodo aparecerá en la categoría `image/logo`

## Inputs

| Input | Tipo | Descripción |
|-------|------|-------------|
| `image` | IMAGE | Imagen base (prenda) |
| `logo` | IMAGE | Imagen del logo a transferir |
| `mask` | MASK | Máscara que define dónde colocar el logo |
| `scale_mode` | STRING | Modo de escalado: "fit", "fill", "stretch" |
| `opacity` | FLOAT | Opacidad del logo (0.0 a 1.0) |

## Outputs

| Output | Tipo | Descripción |
|--------|------|-------------|
| `result_image` | IMAGE | Imagen resultado con el logo aplicado |

## Modos de Escalado

- **fit**: Mantiene proporción, ajusta sin cortar
- **fill**: Mantiene proporción, llena el área (puede cortar)
- **stretch**: Estira para llenar exactamente el área

## Workflow de Ejemplo

1. Carga una imagen de prenda
2. Carga la imagen del logo
3. Crea o carga una máscara que defina dónde va el logo
4. Conecta todo al nodo Logo Transfer
5. Ajusta opacidad y modo de escalado según necesites

## Requisitos

- ComfyUI
- PyTorch
- PIL (Pillow)
- NumPy
- OpenCV (cv2)

## Troubleshooting

### El logo no aparece
- Verifica que la máscara no esté vacía
- Asegúrate de que la máscara tenga valores > 0 en el área deseada

### El logo se ve distorsionado
- Prueba diferentes modos de escalado
- Verifica que la máscara tenga forma apropiada

### Error de dimensiones
- Asegúrate de que todas las imágenes tengan dimensiones compatibles
- La máscara debe corresponder al área deseada en la imagen base

## Versión

1.0.0 - MVP inicial

## Licencia

MIT License
