import os
import json

# Configuración de rutas (Ajusta estas rutas si es necesario)
JSON_ORIGINAL_PATH = "../../data/emoart-5k/images-selected/annotations-full-data.json"  # Ruta a tu JSON con las descripciones
IMAGES_ROOT_DIR = "../classifier/emoart-gan-train/final-images"  # Ruta raíz de tus imágenes filtradas
OUTPUT_METADATA_FILE = os.path.join(IMAGES_ROOT_DIR, "metadata.jsonl")

# Mapeo de carpetas de estilo a su respectivo Trigger Token único
STYLE_TRIGGERS = {
    "color-field-painting": "clrfld style",
    "cubism": "cbsm style",
    "impressionism": "imprsn style",
    "pop-art": "popart style"
}


def load_json_annotations(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_description_lookup(json_data):
    """
    Crea un diccionario optimizado donde la clave es el nombre del archivo (ej: '0006080_YuriZlotnikov-Signalseries.jpg')
    y el valor es el texto descriptivo estructurado.
    """
    lookup = {}
    for entry in json_data:
        raw_path = entry.get("image_path", "")
        if not raw_path:
            continue

        filename = os.path.basename(raw_path.replace("\\", "/"))

        try:
            description_text = entry["description"]["first_section"]["description"]
            lookup[filename] = description_text
        except KeyError:
            continue
    return lookup


def generate_lora_metadata():
    print("Cargando anotaciones del archivo JSON...")
    json_data = load_json_annotations(JSON_ORIGINAL_PATH)
    description_map = build_description_lookup(json_data)

    metadata_lines = []
    matched_count = 0
    missing_desc_count = 0

    print("Escaneando carpetas de imágenes locales...")
    for style_folder, trigger_token in STYLE_TRIGGERS.items():
        folder_path = os.path.join(IMAGES_ROOT_DIR, style_folder)

        if not os.path.exists(folder_path):
            print(f"Advertencia: La carpeta {folder_path} no existe. Saltando...")
            continue

        for img_filename in os.listdir(folder_path):
            if not img_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            if img_filename in description_map:
                raw_description = description_map[img_filename]

                final_prompt = f"{trigger_token}, {raw_description}"

                relative_image_path = f"{style_folder}/{img_filename}"

                metadata_entry = {
                    "file_name": relative_image_path,
                    "text": final_prompt
                }

                metadata_lines.append(json.dumps(metadata_entry, ensure_ascii=False))
                matched_count += 1
            else:
                missing_desc_count += 1

    with open(OUTPUT_METADATA_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(metadata_lines))

    print("\n--- Proceso Completado ---")
    print(f"Imágenes correctamente procesadas y asociadas: {matched_count}")
    print(f"Imágenes encontradas localmente pero sin descripción en el JSON: {missing_desc_count}")
    print(f"Archivo de metadatos creado exitosamente en: {OUTPUT_METADATA_FILE}")


if __name__ == "__main__":
    generate_lora_metadata()