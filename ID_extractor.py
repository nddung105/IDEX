"""
Author: Louis Do
"""
import numpy as np
import cv2
from scipy.spatial import distance as dist
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class RotationChecker:
    def __init__(self):
        pass


    def reset_graph(self,seed=42):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)

    def CheckCardRotation(self,im):
        self.reset_graph()
        self.saver = tf.train.import_meta_graph("./saved_models/flipping_model/my_model_final_flipping.ckpt.meta")
        self.X = tf.get_default_graph().get_tensor_by_name('X:0')
        self.logits = tf.get_default_graph().get_tensor_by_name('fc/outputs/BiasAdd:0')
        self.init = tf.global_variables_initializer()
        with tf.Session(config=config) as sess:
            sess.run(self.init)
            self.saver.restore(sess, "./saved_models/flipping_model/my_model_final_flipping.ckpt")
            y_prob = tf.nn.softmax(sess.run(self.logits, {self.X: im.reshape(1, 512, 512, 3)}))
            y_pred = np.argmax(y_prob.eval(), axis=1)

        rotation_stat = y_pred[0]
        if rotation_stat == 0:
            return cv2.resize(im, (320, 200))
        elif rotation_stat == 1:
            M = cv2.getRotationMatrix2D((256, 256), 180, 1)
        elif rotation_stat == 2:
            M = cv2.getRotationMatrix2D((256, 256), 90, 1)
        elif rotation_stat == 3:
            M = cv2.getRotationMatrix2D((256, 256), -90, 1)

        return cv2.resize(cv2.warpAffine(im, M, (512, 512)), (320, 200))


class IDex:
    def __init__(self):
        self.corners=None
        self.rotchecker = RotationChecker()

    def load(self,path):
        self.im=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

    def _find_canny_thresholds(self,im):
        """
        Find lower and upper threshold for canny edge detection algorithm
        Credit: PyImageSearch
        """
        v = np.median(im)
        sigma = 0.33
        lower_thresh = int(max(0, (1.0 - sigma) * v))
        upper_thresh = int(min(255, (1.0 + sigma) * v))
        return lower_thresh, upper_thresh

    def adjust_gamma(self,image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)

    def unsharp_masking(self,im, ksize=5):
        gaussian = cv2.GaussianBlur(im, (ksize, ksize), 0)
        res = cv2.addWeighted(im.copy(), 1.5, gaussian, -0.5, 0, im.copy())
        return res

    def find_corners(self,im,mode='rgb'):
        assert mode in ['rgb','h','s','hs'], 'wrong mode'
        def order_points(pts):
            xSorted = pts[np.argsort(pts[:, 0]), :]

            leftMost = xSorted[:2, :]
            rightMost = xSorted[2:, :]

            leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
            (tl, bl) = leftMost

            D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
            (br, tr) = rightMost[np.argsort(D)[::-1], :]

            tl -= 2
            br += 2
            tr[0] += 2
            tr[1] -= 2
            bl[0] -= 2
            bl[1] += 2

            return np.array([tl, tr, bl, br], dtype="float32")




        img = cv2.pyrMeanShiftFiltering(im, 10, 40, 3)
        if mode == 'rgb': blur = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        elif mode == 'h':
            img = self.unsharp_masking(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            blur = cv2.normalize(img[:,:,0].astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        elif mode=='s':
            img = self.unsharp_masking(img)
            img = cv2.cvtColor(img[:,:,1], cv2.COLOR_RGB2HSV)
            blur = cv2.normalize(img[:, :, 0].astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        else:
            img = self.unsharp_masking(img)
            img = cv2.cvtColor(img[:, :, 1], cv2.COLOR_RGB2HSV)
            blur = cv2.normalize(img[:, :, :2].astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        blur = cv2.GaussianBlur(blur, (9, 9), 1.6)
        blur = blur.astype(np.uint8)
        low, up = self._find_canny_thresholds(blur)
        edge = cv2.Canny(blur, low, up)
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, None)
        low, up = self._find_canny_thresholds(edge)
        edge = cv2.Canny(edge, low, up)
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, None)
        cnts = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        for c in cnts:
            hull = cv2.convexHull(c)
            cv2.drawContours(edge, [hull], -1, 255, 3)






        cnts = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        our_cnt = np.array([])
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
            if len(approx) == 4:
                our_cnt=None
                our_cnt = approx
                break
        if len(our_cnt)==0:
            self.corners=our_cnt
            return

        self.corners=order_points(our_cnt.reshape(-1,2))

    def detect(self,mode):
        holder_shape = (512, 512)
        holder_height = holder_shape[0]
        holder_width = holder_shape[1]

        self.find_corners(self.im, mode)

        try:
            p = np.array([[0, 0], [holder_width - 1, 0], [0, holder_height - 1], [holder_width - 1, holder_height - 1]],
                             np.float32)
            M = cv2.getPerspectiveTransform(self.corners.astype(np.float32), p)
            projected = cv2.warpPerspective(self.im, M, (512, 512))
            res = self.rotchecker.CheckCardRotation(projected)
            # unsharp masking
            gaussian = cv2.GaussianBlur(res, (7, 7), 10)
            res = cv2.addWeighted(res, 1.5, gaussian, -0.5, 0, res)

        except Exception:
            return None

        return res

