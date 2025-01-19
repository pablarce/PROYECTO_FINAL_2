# src/train.py
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from preprocessing import preprocess_image
from model import build_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_data(data_dir):
    """
    Carga el dataset desde una estructura con subcarpetas por clase.
    """
    X, y = [], []
    class_labels = {}  # Diccionario para mapear clases a índices
    current_label = 0

    for category in ['MAYUS', 'MINUS', 'NUMBERS']:
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            print(f"Advertencia: La categoría '{category}' no existe en {data_dir}")
            continue

        for class_name in os.listdir(category_path):
            class_path = os.path.join(category_path, class_name)
            if os.path.isdir(class_path):
                # Añadir la clase al diccionario de etiquetas si no existe
                if class_name not in class_labels:
                    class_labels[class_name] = current_label
                    current_label += 1

                # Cargar las imágenes de la clase
                for filename in os.listdir(class_path):
                    image_path = os.path.join(class_path, filename)
                    try:
                        image = preprocess_image(image_path)
                        X.append(image)
                        y.append(class_labels[class_name])
                    except Exception as e:
                        print(f"Error procesando la imagen {image_path}: {e}")
            else:
                print(f"Advertencia: '{class_path}' no es un directorio válido.")

    X = np.array(X).reshape(-1, 32, 32, 1) / 255.0  # Normalización
    y = np.array(y)
    return X, y, class_labels

if __name__ == "__main__":
    # Directorios de entrenamiento y validación
    train_dir = "data/dataset/train/"
    validation_dir = "data/dataset/validation/"

    # Cargar datos
    X_train, y_train, class_labels = load_data(train_dir)
    X_val, y_val, _ = load_data(validation_dir)

    print(f"Clases detectadas: {class_labels}")

    # Crear y entrenar el modelo
    model = build_model(input_shape=(32, 32, 1), num_classes=len(class_labels))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

    # Crear la carpeta de resultados si no existe
    os.makedirs("results", exist_ok=True)

    # Guardar el modelo entrenado
    model.save("results/model.h5")
    print("Modelo guardado en 'results/model.h5'")

    # Guardar las etiquetas de las clases
    np.save("results/class_labels.npy", class_labels)
    print("Etiquetas de clases guardadas en 'results/class_labels.npy'")
