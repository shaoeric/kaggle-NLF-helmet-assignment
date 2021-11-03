import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net

# resize and keep ratio: from Shaobt and try to get to 128 * 64
def resize_keep_ratio_v2(img, size=(128, 64)):	
    h, w = img.shape[0], img.shape[1]
    fixed_side_h, fixed_side_w = size
    scale = max(w / float(fixed_side_w), h/float(fixed_side_h))
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (fixed_side_h - new_h) // 2, (fixed_side_h - new_h) // 2, (fixed_side_w - new_w) // 2 + 1, (
            fixed_side_w - new_w) // 2
    elif new_w % 2 == 0 and new_h % 2 != 0:
        top, bottom, left, right = (fixed_side_h - new_h) // 2 + 1, (fixed_side_h - new_h) // 2, (fixed_side_w - new_w) // 2, (
            fixed_side_w - new_w) // 2
    elif new_w % 2 == 0 and new_h % 2 == 0:
        top, bottom, left, right = (fixed_side_h - new_h) // 2, (fixed_side_h - new_h) // 2, (fixed_side_w - new_w) // 2, (
            fixed_side_w - new_w) // 2
    else:
        top, bottom, left, right = (fixed_side_h - new_h) // 2 + 1, (fixed_side_h - new_h) // 2, (fixed_side_w - new_w) // 2 + 1, (
            fixed_side_w - new_w) // 2

    pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])        
    return pad_img


class Extractor(object):
    def __init__(self, model_path="./8_28_helmet_ckpt.t7", use_cuda=True, num_class=751, size=(128, 64), keep_ratio=False):
        self.net = Net(reid=True, num_classes=num_class)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

        state_dict = torch.load(model_path, map_location=torch.device(self.device))['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = size
        self.keep_ratio = keep_ratio
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (128,64) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        def _resize(im, size):
            if not self.keep_ratio:
                return cv2.resize(im.astype(np.float32), (size[1], size[0]))
            else:   
                return resize_keep_ratio_v2(im, size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("E:/000000000785.jpg")
    extr = Extractor("./helmet_ckpt_8_29.t7", num_class=1320, keep_ratio=True)
    feature = extr([img])
    print(feature.shape)
