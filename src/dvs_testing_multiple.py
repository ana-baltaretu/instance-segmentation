import os

import numpy as np

from src.dvs_config import *
from src.dvs_dataset_multiple import RGBDDatasetMultiple
import json

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

config = DvsConfig()
config.display()

# Training dataset
# dataset_train = RGBDDataset()
# dataset_train.load('../data/N_MNIST_images_actually_all_10ms', 'training')
# dataset_train.prepare()

# Testing dataset
dataset_validation = RGBDDatasetMultiple()
# dataset_validation.load('../data/N_MNIST_images_actually_all_10ms', 'validation')
# dataset_validation.load('../data/N_MNIST_images_10ms_skip_50', 'validation')
dataset_validation.load('../data/N_MNIST_images_20ms_skip_50', 'validation')
# dataset_validation.load('../data/N_MNIST_images_50ms_skip_50', 'validation')
dataset_validation.prepare()

class InferenceConfig(DvsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, 'temp_logs/__table_15ep_20ms_coco_skip_50', "mask_rcnn_dvs.h5")
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_dvs_2ep.h5")
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_dvs_5ep.h5")
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_dvs.h5")
# model.set_log_dir('temp_logs/__table_15ep_20ms_coco_skip_50')
model_path = model.find_last()
print(model_path)



# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
print("MODEL")
# print(model.config.display())
print('\n\n\n')

for i in range(30):
    # Test on a random image
    image_id = random.choice(dataset_validation.image_ids)
    print(image_id)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_validation, inference_config,
                               image_id)

    print("bb", len(gt_bbox))#, gt_bbox[1:])
    print("mask", len(gt_mask))#, gt_mask[1:])
    print("id", gt_class_id[1:])
    print("class", dataset_validation.class_names[1:])

    visualize.display_instances(original_image, gt_bbox[1:], gt_mask[1:], gt_class_id[1:],
                                dataset_validation.class_names[1:], figsize=(8, 8))

    results = model.detect([original_image], verbose=0)

    r = results[0]
    # print("How many masks:", len(results))
    # print(r['class_ids'])
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_validation.class_names, scores=r['scores'])

# # Compute VOC-Style mAP @ IoU=0.5
# image_ids = dataset_validation.image_ids  # np.random.choice(dataset_validation.image_ids, 500)
image_ids = np.random.choice(dataset_validation.image_ids, 500)
APs, ACCs, IoUs = [], [], []
for image_id in image_ids:
    # print(image_id)
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_validation, inference_config,
                               image_id)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]

    # Compute AP
    AP, precisions, recalls, overlaps, ious = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=0)
    # print(AP, overlaps)
    print(ious)
    accuracy = utils.compute_accuracy(r['masks'], gt_mask)
    # print(accuracy)
    if len(ious) > 0:
        IoUs.append(np.mean(ious))
    else:
        IoUs.append(0.0)
    APs.append(AP)
    ACCs.append(accuracy)

print("mean AP: ", np.mean(APs))
print("mean IoUs: ", np.mean(IoUs))
print("mean Accuracies: ", np.mean(ACCs))

# visualize.display_differences()