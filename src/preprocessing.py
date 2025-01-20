import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Carga y preprocesa una imagen.
    - Conversión a escala de grises
    - Binarización
    - Ajuste de tamaño
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    resized = cv2.resize(binary, (32, 32), interpolation=cv2.INTER_AREA)
    return resized
