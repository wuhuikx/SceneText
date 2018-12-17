import os
import cv2
import numpy as np
import tensorflow as tf

import model
from icdar import restore_rectangle
import locality_aware_nms as nms_locality

# param name, value, description
tf.app.flags.DEFINE_string('test_data_path', './demo_images/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '../EAST/tmp/east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', './output/', '')
tf.app.flags.DEFINE_string('batch_size', '1', '')

FLAGS = tf.app.flags.FLAGS

def get_batch_images(image_path):
    files = []
    batch_files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(image_path):
        print("---parent = {},{},{}".format(parent, dirnames, filenames))
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    print("----filename = {}".format(os.path.join(parent, filename)))
                    batch_files.append(os.path.join(parent, filename))
                    break
            if len(batch_files) == int(FLAGS.batch_size):
                files.append(batch_files)
                batch_files = []
    print("----len(files) = {}".format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    h, w, _ = im.shape
    if max(h, w) > max_side_len:
        ratio = float(max_side_len) / h if h > w else float(max_side_len) / w
    else:
        ratio = 1
    resize_h = int(ratio * h)
    resize_w = int(ratio * w)

    if resize_h % 32 !=0:
        resize_h = (resize_h // 32 - 1) * 32
    if resize_w % 32 !=0:
        resize_w = (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    print("---im_resize = {}".format(im.shape))
    print("---im_resize = {}".format(im))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, ratio_h, ratio_w
    
def mean_substraction(image, means=[123.68, 116.78, 103.94]):
    num_channel=image.shape[2]
    for i in range(num_channel):
        image[:,:,i] -= np.ones([image.shape[0], image.shape[1]], dtype = np.float32) *means[i]
    return image


def preprocess_image(batch_image_names):
    h_max = 0
    w_max = 0
    batch_images = []
    batch_image_mean_sub = []
    print("length_batch_image_names={}".format(len(batch_image_names)))

    for i,im_name in enumerate(batch_image_names):
        im = cv2.imread(im_name)[:, :, ::-1]
        print("----im = {}".format(im))

        batch_images.append(im)
        im, ratio_h, ratio_w = resize_image(im)
        h_max = h_max if im.shape[0] < h_max else im.shape[0]
        w_max = w_max if im.shape[1] < w_max else im.shape[1]
        print("---im_resize = {}".format(im))
        batch_image_mean_sub.append(im)

    print("length_batch_image_sub={}".format(len(batch_image_mean_sub)))
    
    for i, im in enumerate(batch_image_mean_sub):
        im = mean_substraction(im.astype(np.float32))
        print("---im_sub = {}".format(im))
        image = np.zeros([h_max, w_max, 3], np.float32)
        image[0:im.shape[0],0:im.shape[1],:] = im
        batch_image_mean_sub[i] = image
    print("length_batch_image_sub2={}".format(len(batch_image_mean_sub)))

    return batch_images, batch_image_mean_sub, ratio_h, ratio_w    
        

def postprocess_image(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thresh=0.2):
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]

    # filter by the score map
    xy_text = np.argwhere(score_map > score_map_thresh) 
    xy_text = xy_text[np.argsort(xy_text[:, 0])] # sort the text boxes via the y axis

    # filter by the nms
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4,
                geo_map[xy_text[:, 0],
                xy_text[:, 1], :])
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thresh)
    print("num_boxes before nms = {}".format(text_box_restored.shape[0]))
    print("num_boxes after nms = {}".format(boxes.shape[0]))

    if boxes.shape[0] == 0:
        return None

    # filter low score boxes by the average score map (different from the original paper)
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1,4,2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:,8] > box_thresh]

    return boxes


def save_image(batch_image, boxes, output_dir, image_name):

    def sort_poly(box):
        min_axis = np.argmin(np.sum(box, axis=1))
        box = box[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
        if abs(box[0,0] - box[1,0]) > abs(box[0,1] - box[1,1]):
            return box
        else:
            return box[[0, 3, 2, 1]]

    print("---boxes = {}".format(boxes))
    for i, im_name in enumerate(image_name):
        box_file = os.path.join(FLAGS.output_dir,
          '{}.txt'.format(os.path.basename(im_name).split('.')[0]))
        
        with open(box_file, 'w') as f:
            for box in boxes:
                print("---box = {}".format(box))
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                f.write('{},{},{},{},{},{},{},{}\r\n'.format(box[0,0], box[0,1], box[1,0], box[1,1], box[2,0], box[2,1], box[3,0], box[3,1]))
                im = batch_image[i]
                print("---im= {}".format(im))
                print("---im_shape={}".format(im.shape))
                print("---box = {}".format(box))
                cv2.polylines(im[:,:,::-1], [box.astype(np.int32).reshape((-1,1,2))], True, color=(255,255,0), thickness=1)

        if output_dir:
            img_path = os.path.join(output_dir, os.path.basename(im_name))
            cv2.imwrite(img_path, im[:,:,::-1])


def main(argv=None):
   
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32,
                shape=[None, None, None, 3],
                name='input_images')
        global_step = tf.get_variable('global_step', [], 
                initializer=tf.constant_initializer(0), trainable=False)
        f_score, f_geometry = model.model(input_images, is_training=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path,
                    os.path.basename(ckpt_state.model_checkpoint_path))

            # use this to load the lastest checkpoint 
            # model_file = tf.train.latest_checkpoint('ckpt/')
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            print("--------get_batch_image")
            batch_image_names = get_batch_images(FLAGS.test_data_path)
            for i, one_batch_name in enumerate(batch_image_names):
    
                print("----image_name = {}".format(one_batch_name))
                #-----------preprocessing-----------#
                print("-------begin preprocessing------")
                batch_images, batch_image_mean_sub, ratio_h, ratio_w = preprocess_image(one_batch_name)

                print("-----input_data = {}".format(batch_image_mean_sub))
                print("-----input_data = {}".format(batch_image_mean_sub[0].shape))
                #timer = {'net': 0, 'restore': 0, 'nms': 0}
                #start_net = time.time()
                #-----------run the model-----------#
                print("-------begin inference------")
                score, geometry = sess.run([f_score, f_geometry],
                        feed_dict={input_images:batch_image_mean_sub})
                #timer['net'] = time.time() - start_net

                print("-----score = {}".format(score))
                print("-----geometry = {}".format(geometry))

                #-----------postprocessing-----------#
                print("-------begin postprocessing------")
                boxes = postprocess_image(score_map=score, geo_map=geometry)
                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                #-------------save file--------------#
                print("-------begin save file------")
                print("---image_name = {}".format(one_batch_name[0]))
                if boxes is not None:    
                    save_image(batch_images, boxes, FLAGS.output_dir, one_batch_name)

                print("-------Done!------")

if __name__ == '__main__':
    tf.app.run()
