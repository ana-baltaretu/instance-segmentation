import tonic.transforms as transforms
from label_generation import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == '__main__':
    dataset = tonic.datasets.NMNIST(save_to='../data', train=False)

    #                0,  1,    2,    3,    4,    5,    6,    7,    8,    9
    target_arrays = [0, 1000, 3000, 4000, 4900, 5500, 6500, 7000, 8500, 9200]
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=3)

    # for i in target_arrays:
    #     generate_masks(dataset[i])
    generate_masks(dataset[9200])

    frames_path = 'frames/'
    if not os.path.isdir(frames_path):
        os.mkdir(frames_path)

    file_paths = os.listdir(frames_path)
    images = []
    for image_path in file_paths:
        image = cv2.imread(frames_path + image_path, cv2.IMREAD_COLOR)
        images.append(image)
    generate_gif('./mask_applied.gif', images)

















