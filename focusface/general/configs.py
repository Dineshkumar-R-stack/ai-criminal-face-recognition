AVAILABLE_RESOLUTIONS = {
    'FHD': (1080, 1920, 3),
    'HD': (720, 1280, 2),
    'sHD': (360, 640, 1),
    'VGA': (480, 640, 1),
}

cfg_opt_dict = {
    'data': 'D:/Mini_project/focus_face/data',
    'vid-res': 'sHD',           # good balance for GPU
    'vid-path': 'test/vid_final',
    'label-path': 'test/label_final',
    'out-path': 'output/',
    'det-model': 're50',       # best accuracy (ResNet50 RetinaFace)
    'det-weight': 'weights/Resnet50_Final.pth',
    'box-ratio': 1.30,
    'down': 1,                 # VERY IMPORTANT (speed boost)
    'conf-thresh': 0.55,       # face detection threshold
    'suspect-db': 'seed/suspect_db.csv',
    's-faces': 'target/faces',
    'n-faces': 0,
    'idt-model': 'small',
    'iou-thresh': 0.30,
    'insense': 0.05,             # suspicion threshold
    'criteria': 6.0,           # recognition sensitivity
    'redis-port': 6379,
    'output': 'opencv'
}

qt2opt_vid_res = {
    'adaptive (initialized by data)': 'adaptive',
    'VGA (640x480)': 'VGA',
    'sHD (640x360)': 'sHD',
    'HD (1280x720)': 'HD',
    'FHD (1920x1080)': 'FHD'
}

qt2opt_det_model = {
    'Resnet50_pretrained_RetinaFace': 're50',
    'mobilenet0.25_pretrained_RetinaFace': 'mnet'
}

qt2opt_idt_model = {
    'Small (using 5 Landmarks)': 'small',
    'Large (using 68 Landmarks)': 'large'
}

combo_matcher = {
    'vid-res': qt2opt_vid_res,
    'det-model': qt2opt_det_model,
    'idt-model': qt2opt_idt_model
}
