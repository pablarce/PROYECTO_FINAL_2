import os

# Constantes
CARPETA_IMAGENES = os.path.dirname(__file__)  
SUFIJO = "_imagen28"

def renombrar_imagenes():
    try:
        archivos = os.listdir(CARPETA_IMAGENES)

        for archivo in archivos:
            ruta_completa = os.path.join(CARPETA_IMAGENES, archivo)

            if os.path.isfile(ruta_completa) and archivo.lower().endswith((".jpg", ".png", ".bmp", ".tiff")):
                nombre, extension = os.path.splitext(archivo)
                
                nuevo_nombre = f"{nombre}{SUFIJO}{extension}"
                nueva_ruta = os.path.join(CARPETA_IMAGENES, nuevo_nombre)

                os.rename(ruta_completa, nueva_ruta)
                print(f"Renombrado: {archivo} -> {nuevo_nombre}")

        print("Renombrado completado.")

    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    renombrar_imagenes()
