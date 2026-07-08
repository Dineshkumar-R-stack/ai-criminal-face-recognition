from .models import EmbeddingLoader, FaceComparer
import numpy as np
import cv2


AVAILABLE_RESOLUTIONS = {
    'FHD': (1080, 1920, 3),
    'HD': (720, 1280, 2),
    'sHD': (360, 640, 1),
    'VGA': (480, 640, 1),
}


def box_adapt(boxes: np.ndarray, res: tuple, rat=1.):

    res_boxes = boxes[:, :4] * np.array(res[:2][::-1] + res[:2][::-1])
    boxes_shape = res_boxes[:, 2:] - res_boxes[:, :2]
    boxes_center = (boxes_shape / 2) + res_boxes[:, :2]
    pad_boxes_rad = (boxes_shape * rat if rat != 1. else boxes_shape) / 2.

    pad_res_boxes = np.concatenate(
        (boxes_center - pad_boxes_rad, boxes_center + pad_boxes_rad, boxes[:, 4:]),
        axis=1,
        dtype='float32'
    )

    return pad_res_boxes


class Identifier:

    def __init__(self, embed_db_path: str, n: int, idt_res: str, box_ratio: float, model: str = 'small'):

        self._ebd_loader = EmbeddingLoader(embed_db_path, idt_model=model, n_faces=n)

        self.ebd_dict = self._ebd_loader.ebd_dict
        self.comparer = FaceComparer(self.ebd_dict, idt_model=model)

        self.faces = []
        self.res = AVAILABLE_RESOLUTIONS[idt_res]
        self.box_ratio = box_ratio


    def run(self, img_idt, boxes):

        idt_boxes = []

        h, w = img_idt.shape[:2]
        adapted_boxes = box_adapt(boxes, (h, w, 3), self.box_ratio)

        rgb_frame = img_idt[:, :, ::-1]

        for tmp_box in adapted_boxes:

            tid = -1

            if len(tmp_box) > 5:
                tid = int(tmp_box[5])

            box = [int(b) for b in tmp_box[:4]]
            score = tmp_box[4]

            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(img_idt.shape[1], box[2])
            box[3] = min(img_idt.shape[0], box[3])

            x1, y1, x2, y2 = box

            face_location = [(y1, x2, y2, x1)]

            face_id, face_dist, face_std_score = self.comparer.compare_face(
                rgb_frame,
                face_location,
                get_score=True
            )

            print("IDENTIFIER RESULT:", face_id, face_dist)

            idt_boxes.append([box, score, tid, face_id, face_dist, face_std_score])

        return idt_boxes


    def get_df(self):

        return self._ebd_loader.df