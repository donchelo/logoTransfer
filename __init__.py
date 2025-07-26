from .logo_transfer import LogoTransferNode

# Mapeo de clases de nodos
NODE_CLASS_MAPPINGS = {
    "LogoTransferNode": LogoTransferNode,
}

# Mapeo de nombres para mostrar en la UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "LogoTransferNode": "Logo Transfer",
}

# Información del paquete
__version__ = "1.0.0"
__author__ = "Custom Node Developer"
__description__ = "Custom node para transferir logos a prendas usando máscaras"

# Lista de todos los nodos disponibles
__all__ = ["LogoTransferNode"]

print(f"Logo Transfer Node v{__version__} cargado exitosamente")
