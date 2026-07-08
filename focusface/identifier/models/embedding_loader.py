import hashlib
import pandas as pd
import numpy as np
import random
import face_recognition
import os
db_column_configs = {
    'ID': 'object',
    'NAME': 'object',
    'SEX': 'object',
    'BIRTH': 'object',
    'RAP': 'object',
    'DETAIL': 'object',
    'MUGSHOT_PATH': 'object',
    'IS_TARGET': 'bool'
}
image_ext_list = ['jpg', 'png','jpeg']

class EmbeddingLoader:
    def __init__(self, embed_db_path: str = '', idt_model: str = 'small', n_faces: int = 0):
        self.ebd_dict = {}  # destination
        if os.path.isfile(embed_db_path):
            self.df = pd.read_csv(embed_db_path, dtype=db_column_configs)  
        else:
            print('[ERROR] Can Not Find Embedding DB CSV File')
            exit()
        self.df = self.df.fillna('N/A')  
        self.df.insert(self.df.shape[1], 'CHECKSUM', [False] * self.df.shape[0], True)
        self._model = idt_model
        self._check_db()  
        if n_faces:  # n_faces 입력받았으면
            target_number = self.df[self.df['IS_TARGET']].shape[0]
            if n_faces < target_number or n_faces > self.df.shape[0]:
                # n_faces가 이상하게 세팅되어 있으면 타깃 수와 일치하도록 수정합니다.
                n_faces = target_number
                print(f"[NOTICE] n_face changed as number of target faces: {target_number}")
            selected = random_combination(self.df[~(self.df['IS_TARGET'])].T.to_dict().items(), n_faces)
            self.df = self.df[self.df['IS_TARGET']].append([i[1] for i in selected], ignore_index=True)
        else:
            pass
        self._regen_embeddings()  
        self._read_n_ebd() 
    def _check_db(self):
        for idx, row in self.df.iterrows():
            self.df.at[idx, 'CHECKSUM'] = sha256_checksum(row['MUGSHOT_PATH'], self._model)
    def _regen_embeddings(self):
        print("Generating(Checking) Face Embeddings...", end="")
        for cnt in range(3):
            for idx, row in self.df[~(self.df['CHECKSUM'])].iterrows():
                mugshot_path = row['MUGSHOT_PATH']
                file_name, file_dir = (
                    os.path.split(mugshot_path)[-1],
                    os.path.join(*os.path.split(mugshot_path)[:-1])
                )
                person_dir = file_dir
                embeddings = []
                for img_name in os.listdir(person_dir):
                    if img_name.split('.')[-1].lower() not in image_ext_list:
                        continue
                    img_path = os.path.join(person_dir, img_name)
                    try:
                        img = face_recognition.load_image_file(img_path)
                        encodings = face_recognition.face_encodings(img, model=self._model)
                        if len(encodings) > 0:
                            embeddings.append(encodings[0])
                    except Exception:
                        continue
                if len(embeddings) == 0:
                    print(f"[WARNING] No face embeddings found in {person_dir}")
                    continue
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                file_id = file_name[:file_name.rindex('.')]
                ebd_path = os.path.join(
                    file_dir,
                    '.'.join([file_id, self._model, 'ebd'])
                )
                with open(ebd_path, 'w') as ebd:
                    ebd.write(','.join([str(v) for v in avg_embedding.tolist()]))
                sha256_encrypt(mugshot_path, self._model)
            self._check_db()
            if not self.df[~(self.df['CHECKSUM'])].shape[0]:
                break
            else:
                print(f'[NOTICE] retry embedding generation...({cnt}/3)')
        print("Complete!!")
    def _read_n_ebd(self):
        print("Loading Embeddings...", end="")
        for i in self.df['ID'].values:
            mugshot_paths = self.df[self.df['ID'] == i]['MUGSHOT_PATH'].values
            embeddings = []
            for mugshot_path in mugshot_paths:
                ebd_path = '.'.join(mugshot_path.split('.')[:-1] + [self._model + '.ebd'])
                if os.path.isfile(ebd_path):
                    with open(ebd_path, 'r') as e:
                        line = e.read()
                        ebd_arr = np.array(
                            [float(t) for t in line.replace('\n', '').split(',')],
                            dtype=np.float64
                        )
                        embeddings.append(ebd_arr)
            if len(embeddings) > 0:
                avg_embedding = np.mean(embeddings, axis=0)
                self.ebd_dict.update({i: avg_embedding})
        print("Complete!!")

def random_combination(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)
def sha256_checksum(img_path: str, ext: str):
    ebd_path = '.'.join(img_path.split('.')[:-1] + [ext + '.ebd'])
    checksum_path = '.'.join(img_path.split('.')[:-1] + [ext + '.checksum'])
    if os.path.isfile(ebd_path) and os.path.isfile(checksum_path) and os.path.isfile(img_path):
        try:
            with open(img_path, "rb") as f1, open(ebd_path, "rb") as f2, open(checksum_path, "r") as c:
                b1 = f1.read()  # read entire file as bytes
                b2 = f2.read()  # read entire file as bytes
                readable_hash = hashlib.sha256(b1 + b2).hexdigest()
                checksum_hash = c.read()
                return readable_hash == checksum_hash
        except IOError:
            print('[I/O ERROR] sha256 checksum')
    else:
        return False
def sha256_encrypt(img_path: str, ext: str):
    ebd_path = '.'.join(img_path.split('.')[:-1] + [ext + '.ebd'])
    checksum_path = '.'.join(img_path.split('.')[:-1] + [ext + '.checksum'])
    with open(img_path, "rb") as f1, open(ebd_path, "rb") as f2, open(checksum_path, "w") as c:
        b1 = f1.read()  # read entire file as bytes
        b2 = f2.read()  # read entire file as bytes
        checksum_hash = hashlib.sha256(b1 + b2).hexdigest()
        c.write(checksum_hash)

def create_seed(base_path: str, target_faces_dir: str, sample_faces_dir: str, dst_file_name: str = 'seed'):
    print("Generating Suspect DB Seed...", end="")
    target_faces_path = os.path.join(base_path, target_faces_dir)
    sample_faces_path = os.path.join(base_path, sample_faces_dir)
    tdf = pd.DataFrame(columns=list(db_column_configs.keys()))
    f_list_dict = {}
    if os.path.isdir(target_faces_path) and os.path.isdir(sample_faces_path):
        f_list_dict.update({'target': [target_faces_path, os.listdir(target_faces_path)]})
        f_list_dict.update({'sample': [sample_faces_path, os.listdir(sample_faces_path)]})
    else:
        # TODO: 타겟 패스랑 샘플 패스가 없는경우
        pass
    id_counter = 0
    for cls in f_list_dict.keys():
        for f in f_list_dict[cls][1]:
            fd_path = os.path.join(f_list_dict[cls][0], f)
            if not os.path.isdir(fd_path):
                continue
            images = []
            for n in os.listdir(fd_path):
                if n.split('.')[-1].lower() in image_ext_list:
                    images.append(n)
            if len(images) == 0:
                continue
            f_name = images[0]
            f_path = os.path.join(fd_path, f_name)
            new_row = {
                'ID': str(id_counter).zfill(8),
                'NAME': f,
                'MUGSHOT_PATH': f_path,
                'IS_TARGET': cls == 'target',
            }
            tdf = pd.concat([tdf, pd.DataFrame([new_row])], ignore_index=True)
            id_counter += 1
    tdf.to_csv(os.path.join(base_path, dst_file_name), index=False)
    print("Complete!!")
