import pickle
import numpy as np
from PIL import Image
import argparse
import os


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype(np.uint8))
    image_pil.save(image_path)

parser = argparse.ArgumentParser()
parser.add_argument('--pickle_file', type=str, help="path to pickle file")
parser.add_argument("--save_path", type=str, help="path to save the png images")
args = parser.parse_args()

# READING
f = open(args.pickle_file, 'rb')
data = pickle.load(f)
data_size = len(data)
print ('Will process %d images'%data_size)

# save path
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

for img_id in range(data_size):
    # assuming we always have 1 person
    name = data[img_id]['file_name']
    iuv_image_name = name.split('/')[-1].split('.')[0]+ '_iuv.png'
    iuv_name =  os.path.join(args.save_path, iuv_image_name)
    size = np.array(Image.open(name).convert('RGB')).shape
    wrapped_iuv = np.zeros(size)

    print ('Processing %d/%d: %s'%(img_id+1, data_size, iuv_image_name))
    num_instances = len(data[img_id]['scores'])
    if num_instances is 0:
        print ('%s has no person.'%iuv_image_name)
        file_object.write(iuv_image_name+'\n')
    else:
        # get results - process first detected human
        instance_id = 0
        # process highest score detected human
        # instance_id = data[img_id]['scores'].numpy().tolist().index(max(data[img_id]['scores'].numpy().tolist()))
        pred_densepose_result = data[img_id]['pred_densepose'][instance_id]
        bbox_xyxy = data[img_id]['pred_boxes_XYXY'][instance_id]
        i = pred_densepose_result.labels
        uv = pred_densepose_result.uv * 255
        iuv = np.concatenate((np.expand_dims(i, 0), uv), axis=0)
        # 3xhxw to hxwx3
        iuv_arr = np.transpose(iuv, (1, 2, 0))
        # wrap iuv to size of image
        wrapped_iuv[int(bbox_xyxy[1]):iuv_arr.shape[0]+int(bbox_xyxy[1]), int(bbox_xyxy[0]):iuv_arr.shape[1]+int(bbox_xyxy[0]), :] = iuv_arr
        save_image(wrapped_iuv, iuv_name)
