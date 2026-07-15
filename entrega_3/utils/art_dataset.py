import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ArtDataset(Dataset):
    def __init__(self, data_list, transform=None):

        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def prepare_data_splits(root_dir, macro_classes, sub=False):
    """
    Lee las imágenes y prepara sus etiquetas.

    sub=False:
        root_dir/macro_clase/imagen.jpg

    sub=True:
        root_dir/macro_clase/sub_estilo/imagen.jpg
    """
    class_to_idx = {cls: idx for idx, cls in enumerate(macro_classes)}

    all_data = []
    sub_styles = []

    valid_extensions = ('.png', '.jpg', '.jpeg')

    for macro_cat in macro_classes:
        macro_path = os.path.join(root_dir, macro_cat)

        if not os.path.isdir(macro_path):
            continue

        if sub:
            # Estructura: macro_clase/sub_estilo/imagen
            for sub_style in os.listdir(macro_path):
                sub_style_path = os.path.join(macro_path, sub_style)

                if not os.path.isdir(sub_style_path):
                    continue

                for file_name in os.listdir(sub_style_path):
                    if file_name.lower().endswith(valid_extensions):
                        img_path = os.path.join(
                            sub_style_path, file_name
                        )

                        all_data.append(
                            (img_path, class_to_idx[macro_cat])
                        )
                        sub_styles.append(sub_style)

        else:
            # Estructura: macro_clase/imagen
            for file_name in os.listdir(macro_path):
                img_path = os.path.join(macro_path, file_name)

                if (
                    os.path.isfile(img_path)
                    and file_name.lower().endswith(valid_extensions)
                ):
                    all_data.append(
                        (img_path, class_to_idx[macro_cat])
                    )

                    # Para poder estratificar sin subestilos
                    sub_styles.append(macro_cat)

    # Train 70%, Val 15%, Test 15%
    train_data, temp_data, train_sub, temp_sub = train_test_split(
        all_data,
        sub_styles,
        test_size=0.30,
        random_state=42,
        stratify=sub_styles
    )

    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.50,
        random_state=42,
        stratify=temp_sub
    )

    return train_data, val_data, test_data


def plot_art_grid(dataset, macro_classes=['abstract', 'classic', 'fluid', 'graphic']):
    """
    Muestra una cuadrícula de 5x5 imágenes aleatorias del dataset.
    """
    # Fijamos dimensiones de la figura (ancho, alto en pulgadas)
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

    # Obtenemos 25 índices aleatorios sin repetir del tamaño de tu train_dataset
    indices = np.random.choice(len(dataset), 25, replace=False)

    for i, idx in enumerate(indices):
        # Calculamos la posición en la matriz (fila, columna)
        row = i // 5
        col = i % 5

        # Extraemos la imagen (tensor) y su etiqueta numérica del dataset
        img_tensor, label_idx = dataset[idx]

        # 1. Deshacer la permutación de PyTorch (C, H, W) -> (H, W, C)
        img_numpy = img_tensor.permute(1, 2, 0).numpy()

        # 2. Control de rango (Clip) por seguridad
        # plt.imshow espera valores entre [0, 1] si es float.
        # A veces por operaciones numéricas ligeras se pasa un poco, así que lo limitamos.
        img_numpy = np.clip(img_numpy, 0, 1)

        # Pintamos la imagen en su respectivo cuadro
        axes[row, col].imshow(img_numpy)

        # Ponemos de título el nombre de la macro-categoría
        axes[row, col].set_title(macro_classes[label_idx], fontsize=9)

        # Limpiamos los ejes (los números de los píxeles) para que se vea estético
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def export_train_for_gan(train_list, target_dir="emoart-gan-train",
                         macro_classes=['abstract', 'classic', 'fluid', 'graphic']):
    """
    Copia físicamente las imágenes del set de entrenamiento a una nueva carpeta estructurada.
    """
    # Si la carpeta ya existe, la borramos para empezar limpios
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    os.makedirs(target_dir, exist_ok=True)

    # Creamos las 4 subcarpetas macro
    for cls in macro_classes:
        os.makedirs(os.path.join(target_dir, cls), exist_ok=True)

    print("Copiando imágenes del set de Train original...")
    for img_path, label_idx in train_list:
        macro_name = macro_classes[label_idx]

        # Obtenemos solo el nombre del archivo (ej. " Rembrandt_01.jpg")
        file_name = os.path.basename(img_path)

        # Definimos el destino (ej. "emoart-gan-train/classic/Rembrandt_01.jpg")
        destination = os.path.join(target_dir, macro_name, file_name)

        # Copiamos físicamente la imagen
        shutil.copy(img_path, destination)

    print(f"¡Listo! Dataset de Train exportado con éxito a la carpeta física: '{target_dir}'")


def append_synthetic_images(original_train_list, synthetic_root_dir,
                            macro_classes=['color-field-painting', 'cubism', 'impressionism', 'pop-art']):
    """
    Lee las imágenes sintéticas generadas del disco y las une al train_list original.

    original_train_list: Tu lista actual de tuplas [(ruta_real, label_idx), ...]
    synthetic_root_dir: Ruta carpeta principal de imágenes generadas
    macro_classes: Lista con el orden exacto de tus clases actuales (para mapear al ID numérico)
    """
    # Creamos una copia para no alterar la lista original directamente en memoria
    combined_train_list = list(original_train_list)

    # Mapeo de nombre de clase a ID numérico
    class_to_idx = {cls.lower(): idx for idx, cls in enumerate(macro_classes)}

    synthetic_count = 0

    # Recorremos las carpetas de las clases sintéticas
    for class_name in os.listdir(synthetic_root_dir):
        class_path = os.path.join(synthetic_root_dir, class_name)

        # Ignoramos si no es carpeta o si no está en nuestras clases objetivo
        if not os.path.isdir(class_path) or class_name.lower() not in class_to_idx:
            continue

        label_idx = class_to_idx[class_name.lower()]

        # Leemos todos los archivos de imagen en esa carpeta
        for file_name in os.listdir(class_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, file_name)

                # Añadimos la tupla (ruta_sintetica, label) a la lista combinada
                combined_train_list.append((img_path, label_idx))
                synthetic_count += 1

    print(f"¡Fusión completada!")
    print(f"-> Imágenes reales en Train: {len(original_train_list)}")
    print(f"-> Imágenes sintéticas añadidas: {synthetic_count}")
    print(f"-> Total nuevo Train combinado: {len(combined_train_list)}")

    return combined_train_list