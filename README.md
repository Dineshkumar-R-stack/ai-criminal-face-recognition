# AI Criminal Face Recognition

Real-time criminal face recognition system using RetinaFace + Face Recognition.

## Dataset

Overall Test Dataset:  
https://drive.google.com/drive/folders/1GaGQOeaM-UlPuO0eLnFDxrxEJCG7J-V9?usp=sharing

Demo Video:  
https://drive.google.com/file/d/150ASIsI86AY0qonqZwaCsHAXlMc3fEGG/preview

---

# Project Structure

---

# Installation

### 1. Create Python Environment

Recommended Python version: **3.8**

---

### 2. Install Dependencies

---

### 3. Download Model Weights

Download weights and place them inside:

Sources:

RetinaFace:  
https://github.com/biubug6/Pytorch_Retinaface

Face Recognition:  
https://github.com/ageitgey/face_recognition

SORT Tracking:  
https://github.com/abewley/sort

---

# Running the System

Run the webcam recognition system:

The system will:

1. Detect faces
2. Track faces
3. Compare with criminal database
4. Identify suspects in real-time

---

# Configuration

Configuration file:

Example parameters:
cfg_opt_dict = {
'data': '../data',
'vid-res': 'HD',
'det-model': 're50',
'det-weight': 'weights/Resnet50_Final.pth',
'box-ratio': 1.30,
'conf-thresh': 0.50,
'suspect-db': 'seed/suspect_db.csv',
'n-faces': 20,
'idt-model': 'small'
}


---

# Technologies Used

- RetinaFace (Face Detection)
- face_recognition (Face Embedding)
- SORT (Tracking)
- OpenCV
- PyTorch

---

# Citation

RetinaFace:
@inproceedings{deng2020retinaface,
title={Retinaface: Single-shot multi-level face localisation in the wild},
author={Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={5203--5212},
year={2020}
}


Dlib:
@Article{dlib09,
author = {Davis E. King},
title = {Dlib-ml: A Machine Learning Toolkit},
journal = {Journal of Machine Learning Research},
year = {2009},
volume = {10},
pages = {1755-1758},
}