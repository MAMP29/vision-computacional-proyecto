# Python ML Project

Entorno de trabajo para **Machine Learning y Deep Learning** usando:

* Torch + Torchvision
* Polars para manejo de datos
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter Notebooks

Compatible con **Python 3.12**. Probablemente, sirva con el 3.13, pero debe removerse el < 3.13 en el pyproject
si se monta por esos medios.

---

# 1. Clonar o crear el proyecto

```bash
git clone <repo>
cd python-project
```

El proyecto usa configuración basada en `pyproject.toml`.

---

# 2. Instalación con uv (recomendado)

uv es un gestor moderno de paquetes para Python que crea entornos virtuales automáticamente y es mucho más rápido que pip.

Instalar uv:

```bash
pip install uv
```

Instalar dependencias del proyecto:

```bash
uv sync
```

Esto creará automáticamente:

```
.venv/
```

con todas las dependencias instaladas.

Activar el entorno:

Linux / macOS

```bash
source .venv/bin/activate
```

Windows

```powershell
.venv\Scripts\activate
```

---

# 3. Instalación con pip (método clásico)

Crear entorno virtual:

```bash
python -m venv .venv
```

Activar entorno:

Linux / macOS

```bash
source .venv/bin/activate
```

Windows

```powershell
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -e .
```

---

# 4. Usar Jupyter

Iniciar JupyterLab:

```bash
jupyter lab
```

o notebooks clásicos:

```bash
jupyter notebook
```

Si el kernel no aparece:

```bash
python -m ipykernel install --user --name python-project
```

---

# 5. Verificar que Pytorch detecta GPU

Ejecuta en Python:

```bash
python -c "import torch; print(f'¿CUDA?: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Debe aparecer el nombre de la GPU

---

# 6. Estructura del proyecto

```
python-project/
├── entrega_1
│   ├── cnn
│   │   ├── cnn_classification.ipynb
│   │   ├── __init__.py
│   │   └──  model.py
│   ├── __init__.py
│   ├── mlp
│   │   ├── mlp_classification.ipynb
│   │   └── model.py
│   └── utils
│       ├── animal_dataset.py
│       ├── early_stopping.py
│       ├── __init__.py
│       ├── testing.py
│       └── training.py
├── entrega_2
│   ├── two_steps
│   │   ├── __init__.py
│   │   └── two_steps_detection.ipynb
│   └── utils
│       ├── chess_dataset.py
│       ├── __init__.py
│       ├── testing.py
│       └── training.py
├── flake.lock
├── flake.nix
├── pyproject.toml
├── README.md
└── uv.lock
```
# 7. Descargar datos

## Primera entrega
Datos de la primera entrega proporcionados por ultralytics, para descargarlos haga clic [aquí](https://docs.ultralytics.com/datasets/detect/african-wildlife/), poner el zip sobre la carpeta data y extraer

## Segunda entrega
Datos de la segunda entrega proporcionados por RobotFlow, para descargarlos haga clic [aquí](https://universe.roboflow.com/joseph-nelson/chess-pieces-new/dataset/1)
Descargar la version que dice: 416x416auto-orient en formato COCO JSON con el siguiente preprocesado:
- Auto-Orient: Applied
- Resize: Stretch to 416x416

Extraer, copiar los directorios de test, train y valid y pegarlos en la carpeta de data en una subcarpeta llamada
chess-pieces-coco

## Tercera entrega
Datos de la segunda entrega proporcionados por RobotFlow, para descargarlos haga clic [aquí](https://drive.google.com/file/d/1WCBKTyqe2LMAvYBN5Hkl4SG_TZy3Dq1a/view?usp=sharing)
Extraer y copiar en carpeta data