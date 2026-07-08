from .models import cfg_re50, cfg_mnet, PriorBox
from .models.retinaface import *
from .models.utils import *
import torch.backends.cudnn as cudnn
import numpy as np
import os

# torch, cudnn settings
torch.set_grad_enabled(False)
cudnn.benchmark = True


class Detector:

    def __init__(self, weight_path: str, model: str = 're50', conf_thresh: float = 0.5):

        if model == 're50':
            self.cfg = cfg_re50
        elif model == 'mnet':
            self.cfg = cfg_mnet

        # net and model
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = load_model(self.net, os.path.join(weight_path), False)
        self.net.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)
        self.conf_thresh = conf_thresh

    def run(self, img_tensor, vectorized=True, nms_thr: float = 0.4):
        img = np.float32(img_tensor)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        loc, conf, landms = self.net(img)
        priorbox = PriorBox(self.cfg, image_size=tuple(img_tensor.shape[:2]))
        if vectorized:
            priors = priorbox.vectorized_forward()
        else:
            priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_thr)
        dets = dets[keep, :]
        return dets
