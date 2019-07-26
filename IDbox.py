import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from utils import ops as utils_ops


class IDbox:
    def __init__(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile('./saved_models/ID_detection_graph/frozen_inference_graph.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def _load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def run_inference_for_single_image(self,image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[1], image.shape[2])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: image})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.int64)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def divide_into_classes(self,output_dict):
        name_index = []
        dob_index = []
        id_index = []

        for index in range(len(output_dict['detection_classes'])):
            if len(name_index) < 4 and output_dict['detection_classes'][index] == 1:
                name_index.append(index)
            elif len(dob_index) == 0 and output_dict['detection_classes'][index] == 2:
                box = output_dict['detection_boxes'][index]
                if 14 <= abs(int(box[2] * 200) - int(box[0] * 200)) <= 18 and 53 <= abs(
                        int(box[3] * 320) - int(box[1] * 320)) <= 65 and int(box[0] * 200) >= 90 \
                        and int(box[1] * 320) >= 160:
                    dob_index.append(index)
            elif len(id_index) == 0 and output_dict['detection_classes'][index] == 3:
                box = output_dict['detection_boxes'][index]
                if 19 <= abs(int(box[2] * 200) - int(box[0] * 200)) <= 30 and 123 <= abs(
                        int(box[3] * 320) - int(box[1] * 320)) <= 148:
                    id_index.append(index)

        name = {'detection_boxes': output_dict['detection_boxes'][name_index],
                'detection_scores': output_dict['detection_scores'][name_index],
                'detection_classes': output_dict['detection_classes'][name_index]}
        dob = {'detection_boxes': output_dict['detection_boxes'][dob_index],
               'detection_scores': output_dict['detection_scores'][dob_index],
               'detection_classes': output_dict['detection_classes'][dob_index]}
        id = {'detection_boxes': output_dict['detection_boxes'][id_index],
              'detection_scores': output_dict['detection_scores'][id_index],
              'detection_classes': output_dict['detection_classes'][id_index]}

        return name, dob, id

    def get_bbox(self,im, name, name_thresh, id, dob):
        good_names = []
        good_names_idx = []
        if len(name['detection_boxes']) > 0:
            for index in range(len(name['detection_boxes'])):
                if name['detection_scores'][index] > name_thresh:
                    good_name = name['detection_boxes'][index].copy()
                    good_name[0] *= im.shape[0]
                    good_name[2] *= im.shape[0]
                    good_name[1] *= im.shape[1]
                    good_name[3] *= im.shape[1]
                    try:
                        good_names_idx.append(good_name)
                        # good_names.append(im[int(good_name[0]):int(good_name[2]),int(good_name[1]):int(good_name[3])])
                    except Exception:
                        pass
            good_names_idx = sorted(good_names_idx, key=lambda x: x[1])
            for good_name in good_names_idx:
                good_names.append(im[int(good_name[0]):int(good_name[2]), int(good_name[1]):int(good_name[3])])

        if len(id['detection_boxes']) > 0:
            res_id = id['detection_boxes'][0].copy()
            res_id[0] *= im.shape[0]
            res_id[2] *= im.shape[0]
            res_id[1] *= im.shape[1]
            res_id[3] *= im.shape[1]
            good_id = []
            try:
                good_id.append(im[int(res_id[0]):int(res_id[2]), int(res_id[1]):int(res_id[3])])
            except Exception:
                pass

        else:
            good_id = [[]]

        if len(dob['detection_boxes']) > 0:
            res_dob = dob['detection_boxes'][0].copy()
            res_dob[0] *= im.shape[0]
            res_dob[2] *= im.shape[0]
            res_dob[1] *= im.shape[1]
            res_dob[3] *= im.shape[1]
            good_dob = []
            try:
                good_dob.append(im[int(res_dob[0]):int(res_dob[2]), int(res_dob[1]):int(res_dob[3])])
            except Exception:
                pass

        else:
            good_dob = []

        return good_names, good_id, good_dob