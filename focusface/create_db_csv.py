import os
import sys
import pandas as pd
from identifier.models.embedding_loader import create_seed


base = 'D:/Mini_project/focus_face/data'
target_face_path = os.path.join(base, 'target/faces')
sample_face_path = os.path.join(base, 'target/sample')
create_seed(base, target_face_path, sample_face_path,'seed/suspect_db.csv')  # import 되어 있는 코드에서 설명 참고 바랍니다.
