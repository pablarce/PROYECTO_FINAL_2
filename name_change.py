import os

# Constantes
CARPETA_IMAGENES = os.path.dirname(__file__)  # Carpeta del script (misma ubicación)
SUFIJO = "_imagen28"  # Sufijo a añadir a los nombres de las imágenes

# Función principal
def renombrar_imagenes():
    try:
        # Obtener la lista de archivos en la carpeta
        archivos = os.listdir(CARPETA_IMAGENES)

        for archivo in archivos:
            # Construir la ruta completa del archivo
            ruta_completa = os.path.join(CARPETA_IMAGENES, archivo)

            # Verificar que es un archivo y tiene un formato de imagen
            if os.path.isfile(ruta_completa) and archivo.lower().endswith((".jpg", ".png", ".bmp", ".tiff")):
                # Extraer el nombre y la extensión
                nombre, extension = os.path.splitext(archivo)
                
                # Crear el nuevo nombre
                nuevo_nombre = f"{nombre}{SUFIJO}{extension}"
                nueva_ruta = os.path.join(CARPETA_IMAGENES, nuevo_nombre)

                # Renombrar el archivo
                os.rename(ruta_completa, nueva_ruta)
                print(f"Renombrado: {archivo} -> {nuevo_nombre}")

        print("Renombrado completado.")

    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Ejecutar la función
if __name__ == "__main__":
    renombrar_imagenes()
