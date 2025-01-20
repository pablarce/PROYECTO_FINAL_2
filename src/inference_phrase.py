import cv2
import numpy as np
import tensorflow as tf
from preprocessing import preprocess_image
from inference import load_class_labels, predict

def segment_characters(image_path):
    """
    Segmenta los caracteres en una imagen de texto.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV) 

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 15 and w > 5:  
            char_image = binary[y:y+h, x:x+w]
            char_image = cv2.resize(char_image, (32, 32), interpolation=cv2.INTER_AREA)
            char_images.append((x, char_image))

    char_images = sorted(char_images, key=lambda item: item[0])
    return [char[1] for char in char_images]

def infer_phrase(image_path, model_path="results/model.h5", label_path="results/class_labels.npy"):
    """
    Realiza inferencia sobre una palabra o frase completa en una imagen.
    """
    class_labels = load_class_labels(label_path)
    model = tf.keras.models.load_model(model_path)

    char_images = segment_characters(image_path)
    phrase = ""

    for char_image in char_images:
        char_image = char_image.reshape(1, 32, 32, 1) / 255.0 
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
