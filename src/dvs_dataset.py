# https://github.com/orestis-z/mask-rcnn-rgbd
# 2D-3D-S
# https://github.com/orestis-z/mask-rcnn-rgbd/tree/master/instance_segmentation/2D-3D-S
import os, sys
import numpy as np
import skimage.io
import cv2
from src.objects_dataset import ObjectsDataset

from src.dvs_config import *


NAME = "N-MNIST-DVS"


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
               (9, 'Nine')]

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

                    # if not os.path.isfile(depth_path) and not os.path.isfile(mask_path):
                    #     print('Warning: No DEPTH and MASK found for ' + frame_path)
                    # if not os.path.isfile(depth_path):
                    #     print('Warning: No DEPTH found for ' + frame_path)
                    # elif not os.path.isfile(mask_path):
                    print(parent_root)
                    if not os.path.isfile(mask_path):
                        print('Warning: No MASK found for ' + frame_path)
                    else:
                        self.add_image(
                            NAME,
                            image_id=count,
                            path=frame_path,
                            # depth_path=depth_path,
                            mask_path=mask_path,
                            width=self.WIDTH,
                            height=self.HEIGHT,
                            target=number)
                        count += 1

                ind += 1
        print('added {} images for {}'.format(count, subset))

    def load_image(self, image_id, mode="RGBD"):
        """
        TODO: Implement!
        :param image_id:
        :param mode:
        :return:
        """
        image = super(ObjectsDataset, self).load_image(image_id, mode="RGB")
        return image

    def to_mask(img, instance):
        return (img == instance).astype(np.uint8)

    # vectorize since this was slow in serial execution
    to_mask_v = np.vectorize(to_mask, signature='(n,m),(k)->(n,m)')

    def load_mask(self, image_id):
        """
        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        :param image_id:
        :return:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        mask_path = self.image_info[image_id]['mask_path']
        img = cv2.imread(mask_path, -1)

        # https://github.com/alexsax/2D-3D-Semantics/blob/master/assets/utils.py
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        img = R * 256 * 256 + G * 256 + B

        instances = np.unique(img.flatten())
        n_instances = len(instances)

        target = int(self.image_info[image_id]['target']) + 1

        masks = np.repeat(np.expand_dims(img, axis=2), n_instances, axis=2)  # bottleneck code
        masks = self.to_mask_v(masks, instances)
        if not n_instances:
            raise ValueError("No instances for image {}".format(mask_path))

        # class_ids = np.array([1] * n_instances, dtype=np.int32)
        class_ids = np.array([target, 0])
        return masks, class_ids

if __name__ == '__main__':
    dataset = RGBDDataset()
    dataset.load('../data/N_MNIST_images', 'testing')
    dataset.prepare()

    print(dataset.image_ids)
    image_ids = np.random.choice(dataset.image_ids, 4)
    print(image_ids)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        print(dataset.class_names)
        print(class_ids)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=2)
#     print(dataset.image_ids)
#     # masks, class_ids = dataset.load_mask(0)


