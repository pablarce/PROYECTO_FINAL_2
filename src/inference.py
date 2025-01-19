# src/inference.py
import sys
import os
import numpy as np
import tensorflow as tf
from preprocessing import preprocess_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_class_labels(label_path="results/class_labels.npy"):
    """
    Carga el diccionario de etiquetas desde un archivo.
    """
    return np.load(label_path, allow_pickle=True).item()

def predict(image_path, model_path="results/model.h5", label_path="results/class_labels.npy"):
    """
    Realiza una predicción utilizando un modelo previamente entrenado.
    """
    class_labels = load_class_labels(label_path)
    model = tf.keras.models.load_model(model_path)
    
    image = preprocess_image(image_path)
    image = image.reshape(1, 32, 32, 1) / 255.0
    
    predictions = model.predict(image)
    class_idx = tf.argmax(predictions, axis=1).numpy()[0]
    class_name = [k for k, v in class_labels.items() if v == class_idx][0]
    return class_name

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python src/inference.py <ruta_imagen>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    try:
        result = predict(image_path)
        print(f"Predicción para la imagen '{image_path}': {result}")
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
