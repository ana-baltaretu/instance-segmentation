from src.dvs_config import *
from src.dvs_dataset import *

config = DvsConfig()
config.display()

# Training dataset
dataset_train = RGBDDataset()
dataset_train.load('../data/N_MNIST_images', 'training')
dataset_train.prepare()

# Testing dataset
dataset_validation = RGBDDataset()
dataset_validation.load('../data/N_MNIST_images', 'validation')
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
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()



# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

for i in range(20):
    # Test on a random image
    image_id = random.choice(dataset_validation.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_validation, inference_config,
                               image_id) # , use_mini_mask=False

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_train.class_names, figsize=(8, 8))