from src.my_data_generation import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == '__main__':
    # test_dataset = tonic.datasets.NMNIST(save_to='../data', train=False)
    # frames, colorized_masks, target, time_frames = generate_masks(test_dataset[5000])
    # show_events(frames, 'input/frame_' + str(target) + '_')
    # show_events(time_frames, 'input/time_frame_' + str(target) + '_')
    # show_events(colorized_masks, 'frames/label_masked_frame' + str(target) + '_')

    # TODO: uncomment below
    train_dataset = tonic.datasets.NMNIST(save_to='../data', train=True)
    test_dataset = tonic.datasets.NMNIST(save_to='../data', train=False)

    split_train_test_validation('../data/NMNIST', '../data/N_MNIST', cleanup=False, train_data_percentage=0.8)

    generate_rgbd_images_and_masks(train_dataset, test_dataset, '../data/N_MNIST_images_Alex', cleanup=True, skip=1000)
    # TODO: uncomment above

    # #                0,  1,    2,    3,    4,    5,    6,    7,    8,    9
    # target_arrays = [0, 1000, 3000, 4000, 4900, 5500, 6500, 7000, 8500, 9200]
    # sensor_size = tonic.datasets.NMNIST.sensor_size
    # frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=3)

    # for i in target_arrays:
    #     frames, colorized_masks, target = generate_masks(test_dataset[i])
    #     show_events(frames, 'input/frame_' + str(target) + '_')
    #     show_events(colorized_masks, 'frames/label_masked_frame' + str(target) + '_')
    # generate_masks(dataset[9200])

    # frames_path = 'frames/'
    # if not os.path.isdir(frames_path):
    #     os.mkdir(frames_path)
    #
    # file_paths = os.listdir(frames_path)
    # images = []
    # for image_path in file_paths:
    #     image = cv2.imread(frames_path + image_path, cv2.IMREAD_COLOR)
    #     images.append(image)
    # generate_gif('./mask_applied.gif', images)

















