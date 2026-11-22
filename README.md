# IPM - Práctica 2 - Sign language spelling game

![alt text](cover_image.png)


## Introducción

En esta práctica, hemos creado un juego para ayudar a y *gamificar* el aprendizaje del [alfabeto
tactilológico americano (ASL)](https://en.wikipedia.org/wiki/American_Sign_Language).

## Prerequisitos

Tener instalado **Conda**, [instalar aquí](https://www.anaconda.com/docs/getting-started/miniconda/install).

## Requisitos

Crear un entorno de conda:
```bash
conda create -n IPM python=3.12
conda activate IPM
```

## Instalación

Se instalan las dependencias necesarias (MediaPipe, Requests, tqdm, cv2, numpy, etc.):
```bash
pip install -r requirements.txt
```

## Descargar pesos

Script para poder descargar los pesos del modelo *Pose Landmarker*:
```bash
python download_hand_model.py
```


## Ejecución
```bash
python app.py
```

### Ejecución - WSL *passthrough*
Si estás utilizando WSL, antes de poder ejecutar `app.py` tendrás que abrir una ventana de Powershell y ejecutar:
```powershell
usbipd list
```

Identifica la entrada correspondiente a tu camara. Quédate con su `BUSID`.
A continaución, ejecuta:

```powershell
usbipd bind --busid <CAMERA_BUSID>
usbipd attach --wsl --busid <CAMERA_BUSID>
```