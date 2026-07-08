import face_recognition
import numpy as np
import cv2

class FaceComparer:
    """
    Compare detected face embeddings with database embeddings
    and return the closest match.
    """

    def __init__(self, ebd_dict: dict, idt_model: str = 'small', tolerance=0.55):

        self.ebd_dict = ebd_dict
        self.model = idt_model
        self.tolerance = tolerance


    def compare_face(self, frame: np.ndarray, locations, get_score: bool = False):

        face_id = '-'
        match_distance = self.tolerance
        standard_seed = 0.0

        # get underlying dlib encoder
        encoder = face_recognition.api.face_encoder

        try:
            embeddings = []

            for (top, right, bottom, left) in locations:

                face_img = frame[top:bottom, left:right]

                if face_img.size == 0:
                    continue

                # resize to stable encoder size
                face_img = cv2.resize(face_img, (150, 150))

                emb = encoder.compute_face_descriptor(face_img)

                embeddings.append(np.array(emb))

            if len(embeddings) == 0:
                if get_score:
                    return '-', self.tolerance, 0.0
                else:
                    return '-', self.tolerance

        except Exception:
            if get_score:
                return '-', self.tolerance, 0.0
            else:
                return '-', self.tolerance


        face_ebd = embeddings[0]

        db_embeddings = np.array(list(self.ebd_dict.values()))

        face_distances = np.linalg.norm(db_embeddings - face_ebd, axis=1)

        best_match_index = np.argmin(face_distances)
        match_distance = float(face_distances[best_match_index])

        if match_distance <= self.tolerance:
            face_id = list(self.ebd_dict.keys())[best_match_index]

        print("MATCH DIST:", match_distance)

        if get_score:
            return face_id, match_distance, standard_seed
        else:
            return face_id, match_distance