import cv2
import numpy as np
import tensorflow as tf
from preprocessing import preprocess_image
from inference import load_class_labels, predict

def segment_characters(image_path):
    """
    Segmenta los caracteres en una imagen de texto.
    """
    # Leer y binarizar la imagen
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)  # Invertir colores para contornos

    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos por tamaño y ordenarlos de izquierda a derecha
    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 15 and w > 5:  # Filtrar ruido basado en tamaño
            char_image = binary[y:y+h, x:x+w]
            char_image = cv2.resize(char_image, (32, 32), interpolation=cv2.INTER_AREA)
            char_images.append((x, char_image))  # Guardar la posición y la imagen

    # Ordenar los caracteres por la posición horizontal (coordenada x)
    char_images = sorted(char_images, key=lambda item: item[0])
    return [char[1] for char in char_images]

def infer_phrase(image_path, model_path="results/model.h5", label_path="results/class_labels.npy"):
    """
    Realiza inferencia sobre una palabra o frase completa en una imagen.
    """
    # Cargar etiquetas y modelo
    class_labels = load_class_labels(label_path)
    model = tf.keras.models.load_model(model_path)

    # Segmentar caracteres
    char_images = segment_characters(image_path)
    phrase = ""

    # Realizar predicción para cada carácter
    for char_image in char_images:
        char_image = char_image.reshape(1, 32, 32, 1) / 255.0  # Normalización
        predictions = model.predict(char_image)
        class_idx = np.argmax(predictions)
        predicted_char = [k for k, v in class_labels.items() if v == class_idx][0]
        phrase += predicted_char

    return phrase

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Uso: python src/inference_phrases.py <ruta_imagen>")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        result = infer_phrase(image_path)
        print(f"Texto interpretado: {result}")
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
