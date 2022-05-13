# https://tonic.readthedocs.io/en/latest/tutorials/nmnist.html

# https://github.com/neuromorphs/tonic
# https://github.com/tihbe/python-ebdataset
# https://github.com/TimoStoff/event_utils
import tonic
import tonic.transforms as transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dataset = tonic.datasets.NMNIST(save_to='../data', train=False)
my_events, target = dataset[0]
sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=3)
frames = frame_transform(my_events)


if __name__ == '__main__':
    print(target)
    # plot_frames(frames)

    # plot_frames_denoised(frame_transform, my_events)

    ######################

    # denoise_transform = tonic.transforms.Denoise(filter_time=5000)
    # events_denoised = denoise_transform(my_events)
    # positive_event_array = generate_event_arrays(events_denoised, 1)
    # negative_event_array = generate_event_arrays(events_denoised, 0)
    # x_data_pos, y_data_pos, z_data_pos, time_data_pos = positive_event_array
    # x_data_neg, y_data_neg, z_data_neg, time_data_neg = negative_event_array

    ######################

    # plot_1_channel_3D(x_data_pos, y_data_pos, z_data_pos, "Blues", "plots/pos")
    # plot_1_channel_3D(x_data_neg, y_data_neg, z_data_neg, "Reds", "plots/neg")

    # plot_2_channel_3D(x_data_pos, y_data_pos, z_data_pos, "Blues", x_data_neg, y_data_neg, z_data_neg, "Reds", "plots/comb")

    # plot_voxel_grid(my_events, sensor_size)
    # plot_time_surfaces(my_events, sensor_size)
    # show_events(frames, 'frames/fixed_events')
    # show_events(cropped_frames, 'frames/cropped_frames')





