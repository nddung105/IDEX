import numpy as np
import cv2
from IDbox import IDbox
import matplotlib.pyplot as plt
from pytesseract import image_to_string
from correct_name import correct_name
from utils import visualization_utils as vis_util
from utils import label_map_util
import datetime
import codecs
import itertools
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import matplotlib
matplotlib.use('tkagg')


class OCR_helper:
    """
    Just a helper class
    """
    def __init__(self):
        pass

    def _reset_graph(self,seed=42):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)

    def id_num_detect(self,im):

        res = []
        num_list = self.roi(im)


        self._reset_graph()
        self.id_model_saver = tf.train.import_meta_graph("./saved_models/flipping_model/my_model_final_id.ckpt.meta")
        self.id_model_X = tf.get_default_graph().get_tensor_by_name('inputs/X:0')
        self.id_model_logits = tf.get_default_graph().get_tensor_by_name('fc/outputs/BiasAdd:0')

        init = tf.global_variables_initializer()
        with tf.Session(config=config) as sess:
            sess.run(init)
            self.id_model_saver.restore(sess, "./saved_models/flipping_model/my_model_final_id.ckpt")
            for im in num_list:
                y_prob = tf.nn.softmax(sess.run(self.id_model_logits, {self.id_model_X: im.reshape(1, 56, 28, 1)}))
                y_pred = np.argmax(y_prob.eval(), axis=1)
                res.append(str(y_pred[0]))
        return ''.join(res)


    def formatting_dob(self,dob):
        real_dob = []
        for digit in dob:
            if digit.isdigit():
                real_dob.append(digit)
        if len(real_dob)>=8:
            return "{}{}/{}{}/{}{}{}{}".format(real_dob[0], real_dob[1], real_dob[2], real_dob[3], real_dob[4], real_dob[5],
                                           real_dob[6], real_dob[7])
        return dob

    def unsharp_masking(self,im, ksize=5):
        gaussian = cv2.GaussianBlur(im, (ksize, ksize), 0)
        res = cv2.addWeighted(im.copy(), 1.5, gaussian, -0.5, 0, im.copy())
        return res

    def adjust_gamma(self,image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)

    def preprocess_image_dob(self,im):
        res = self.unsharp_masking(cv2.resize(im, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
        res = cv2.pyrMeanShiftFiltering(res, 10, 40, 3)
        return res

    def preprocess_image_ver2(self,im):
        img = cv2.resize(im.copy(), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        img = self.adjust_gamma(img, 1.45)
        img = cv2.pyrMeanShiftFiltering(img, 10, 20, 3)
        img = self.unsharp_masking(img)
        Z = img.reshape((-1, 3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = cv2.cvtColor(res.reshape((img.shape)), cv2.COLOR_BGR2GRAY)
        _, res2 = cv2.threshold(res2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        res2 = 255 - res2
        res2 = cv2.erode(res2, None)
        return res2

    def roi(self,img, itype='id'):
        assert itype.lower() == 'id' or itype.lower() == 'dob'
        im = self.preprocess_image_ver2(cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        cnts = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])[:]
        coords = []
        if itype == 'id':
            for ctr in cnts:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(ctr)
                if 10 <= w and 32 <= h:
                    coords.append((x, y, w, h))
        else:
            for ctr in cnts:
                x, y, w, h = cv2.boundingRect(ctr)
                if 5 <= w and 25 <= h:
                    coords.append((x, y, w, h))

        coords = sorted(coords, key=lambda coord: coord[0])
        res = [cv2.resize(im[coord[1]:coord[1] + coord[3], coord[0]:coord[0] + coord[2]], (28, 56)) for coord in coords]
        return res







class IDOCR:
    """
    IDOCR is used to recognize information from scanned ID card
    """
    def __init__(self):
        self.person_name=None
        self.person_id=None
        self.person_dob=None
        self.IDbox=IDbox()
        self.helper = OCR_helper()

    def reset_attr(self):
        self.person_name = None
        self.person_id = None
        self.person_dob = None

    def _date_to_string(self,dt_time):
        return str(dt_time.year) + str(dt_time.month) + str(dt_time.day) + str(dt_time.hour) + str(
            dt_time.minute) + str(dt_time.second)

    def run(self,im,output_dir=''):
        """
        :param im: Scanned ID card
        :param output_dir: If provided the results will be saved to this directory
        :return There is no returning anything, the result is saved to this class attributes and to 'output_dir' if provided
        """
        self.im=im.copy()
        assert len(output_dir)==0 or os.path.exists(output_dir), "'output_dir' does not exist"

        category_index = label_map_util.create_category_index_from_labelmap('object-detection.pbtxt',
                                                                            use_display_name=True)
        image_np_expanded = np.expand_dims(self.im, axis=0)
        output_dict = self.IDbox.run_inference_for_single_image(image_np_expanded, self.IDbox.detection_graph)
        name, dob, id = self.IDbox.divide_into_classes(output_dict)

        good_names, good_id, good_dob = self.IDbox.get_bbox(self.im, name, 0.9, id, dob)
        full_name = []
        all_full_names=[]
        if len(good_names) > 0:
            for index in range(len(good_names)):
                if output_dir: plt.imsave(output_dir + '\\name{}.jpg'.format(index + 1), good_names[index])
                text = image_to_string(self.helper.unsharp_masking(good_names[index]), config='--psm 13', lang='vie')
                full_name.append(text)

            all_full_names.append(correct_name(full_name[0], 'family_name'))



            for index in range(len(full_name) - 2):
                all_full_names.append(correct_name(full_name[index + 1], 'middle_name'))
            all_full_names.append(correct_name(full_name[-1], 'first_name'))

            self.person_name=list(itertools.product(*all_full_names))
            for index in range(len(self.person_name)):
                self.person_name[index]=' '.join(self.person_name[index]).upper()

            if output_dir:
                f=codecs.open(os.path.join(output_dir,'names.txt'),'w','utf-8')
                f.write('\n'.join(self.person_name))
                f.close()


            vis_util.visualize_boxes_and_labels_on_image_array(
                self.im,
                name['detection_boxes'],
                name['detection_classes'],
                name['detection_scores'],
                category_index,
                instance_masks=name.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=1, skip_labels=True, skip_scores=True, min_score_thresh=0.9)

        if len(good_id) > 0:
            self.person_id=self.helper.id_num_detect(good_id[0])
            if output_dir:
                plt.imsave(output_dir + '\\id.jpg', good_id[0])
                f=codecs.open(os.path.join(output_dir,'id.txt'),'w')
                f.write(self.person_id)
                f.close()

            vis_util.visualize_boxes_and_labels_on_image_array(
                self.im,
                id['detection_boxes'],
                id['detection_classes'],
                id['detection_scores'],
                category_index,
                instance_masks=id.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=1, skip_labels=True, skip_scores=True, min_score_thresh=-1.0)

        if len(good_dob) > 0:
            self.person_dob=self.helper.formatting_dob(image_to_string(good_dob[0],
                                                                       config='--psm 13', lang='vie'))
            if output_dir:
                plt.imsave(output_dir + '\\dob.jpg', good_dob[0])
                f=codecs.open(os.path.join(output_dir,'dob.txt','w'))
                f.write(self.person_dob)
                f.close()

            vis_util.visualize_boxes_and_labels_on_image_array(
                self.im,
                dob['detection_boxes'],
                dob['detection_classes'],
                dob['detection_scores'],
                category_index,
                instance_masks=dob.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=1, skip_labels=True, skip_scores=True, min_score_thresh=-1.0)
        if output_dir: plt.imsave(output_dir + '\\res' + self._date_to_string(datetime.datetime.now()) + ".jpg", self.im)


