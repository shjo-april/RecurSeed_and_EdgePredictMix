import sys
import numpy as np

if sys.platform == 'linux':
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels
    from pydensecrf.utils import unary_from_softmax

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
    
    def __call__(self, image, probmap, bg_th=None, gamma=None):
        C, H, W = probmap.shape

        if bg_th is not None:
            bg = np.ones((1, H, W), dtype=np.float32) * bg_th
            probmap = np.concatenate([bg, probmap], axis=0)

            C += 1

        elif gamma is not None:
            bg = np.power(1 - np.max(probmap, axis=0, keepdims=True), gamma)
            probmap = np.concatenate([bg, probmap], axis=0)

            C += 1
        
        U = unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)
        
        image = np.ascontiguousarray(image)
        
        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))
        
        return Q

    def run_label(self, image, label, num_classes=21, gt_prob=0.7):
        """
        {
            'iter_max':10,
            'bi_w':10, 'bi_xy_std':50, 'bi_rgb_std':5, 
            'pos_w':3, 'pos_xy_std':3
        }
        """
        H, W = label.shape

        U = unary_from_labels(label, num_classes, gt_prob=gt_prob, zero_unsure=False)
        U = np.ascontiguousarray(U)
        
        image = np.ascontiguousarray(image)
        
        d = dcrf.DenseCRF2D(W, H, num_classes)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((num_classes, H, W))
        Q = np.argmax(Q, axis=0)

        return Q