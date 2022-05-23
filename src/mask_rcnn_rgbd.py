# https://github.com/orestis-z/mask-rcnn-rgbd
# 2D-3D-S
# https://github.com/orestis-z/mask-rcnn-rgbd/tree/master/instance_segmentation/2D-3D-S
import os, sys
import numpy as np
import skimage.io
import cv2
from src.objects_dataset import ObjectsDataset


NAME = "N-MNIST_RGBD"


class RGBDDataset(ObjectsDataset):

    WIDTH = 34
    HEIGHT = 34

    CLASSES = [(0, 'Zero'),
               (1, 'One'),
               (2, 'Two'),
               (3, 'Three'),
               (4, 'Four'),
               (5, 'Five'),
               (6, 'Six'),
               (7, 'Seven'),
               (8, 'Eight'),
               (9, 'Nine'),
               (10, 'Background')]

    def load(self, dataset_dir, subset, skip=1):
        assert (subset == 'training' or subset == 'validation' or subset == 'testing')
        dataset_dir = os.path.join(dataset_dir, subset)

        # Add classes
        for cls in self.CLASSES:
            self.add_class(NAME, cls[0] + 1, cls[1])

        ind = 0
        count = 0
        # exclude = set(['depth', 'mask'])
        print(os.listdir(dataset_dir))

        for number in os.listdir(dataset_dir):
            parent_root = os.path.join(dataset_dir, number)
            frame_dir = os.path.join(parent_root, 'frame')
            for frame in os.listdir(frame_dir):
                if ind % skip == 0:
                    frame_id = frame[6:]
                    frame_path = os.path.join(frame_dir, 'frame_' + frame_id)
                    depth_path = os.path.join(parent_root, 'depth', 'depth_' + frame_id)
                    mask_path = os.path.join(parent_root, 'mask', 'mask_' + frame_id)

                    if not os.path.isfile(depth_path) and not os.path.isfile(mask_path):
                        print('Warning: No DEPTH and MASK found for ' + frame_path)
                    if not os.path.isfile(depth_path):
                        print('Warning: No DEPTH found for ' + frame_path)
                    elif not os.path.isfile(mask_path):
                        print('Warning: No MASK found for ' + frame_path)
                    else:
                        self.add_image(
                            NAME,
                            image_id=count,
                            path=frame_path,
                            depth_path=depth_path,
                            mask_path=mask_path,
                            width=self.WIDTH,
                            height=self.HEIGHT)
                        count += 1

                ind += 1
        print('added {} images for {}'.format(count, subset))


if __name__ == '__main__':
    dataset = RGBDDataset()
    dataset.load('../data/N_MNIST_images', 'testing', skip=999)
    print(dataset.image_ids)
    # masks, class_ids = dataset.load_mask(0)


