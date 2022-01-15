import os
import shutil
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, help="path to dataset")
args = parser.parse_args()

IMG_EXTENSIONS = [
'.jpg', '.JPG', '.jpeg', '.JPEG',
'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    train_root = os.path.join(dir, 'train')
    if not os.path.exists(train_root):
        os.mkdir(train_root)

    test_root = os.path.join(dir, 'test')
    if not os.path.exists(test_root):
        os.mkdir(test_root)

    train_images = []
    train_f = open(os.path.join(dir, 'tools', 'train.lst'), 'r')
    for lines in train_f:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            train_images.append(lines)

    test_images = []
    test_f = open(os.path.join(dir, 'tools', 'test.lst'), 'r')
    for lines in test_f:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            test_images.append(lines)


    for root, _, fnames in sorted(os.walk(os.path.join(dir, 'img_highres'))):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                path_names = path.split('/')
                print(path_names)

                path_names = path_names[4:]
                del path_names[1]
                path_names[0] = 'fashion'
                path_names[3] = path_names[3].replace('_', '')
                path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
                path_names = "".join(path_names)

                if path_names in train_images:
                    shutil.copy(path, os.path.join(train_root, path_names))
                    print('saving -- %s'%os.path.join(train_root, path_names))
                elif path_names in test_images:
                    shutil.copy(path, os.path.join(test_root, path_names))
                    print('saving -- %s'%os.path.join(train_root, path_names))

make_dataset(args.dataroot)
