import matplotlib.pyplot as plt
import tonic
import os
from src.my_util import *
from PIL import Image
import tonic.transforms as transforms
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def plot_frames(frames):
    fig, axes = plt.subplots(1, len(frames))
    for axis, frame in zip(axes, frames):
        axis.imshow(frame[1] - frame[0])
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def plot_frames_denoised(frame_transform, events):
    denoise_transform = tonic.transforms.Denoise(filter_time=10000)
    events_denoised = denoise_transform(events)
    frames_denoised = frame_transform(events_denoised)
    plot_frames(frames_denoised)


def plot_1_channel_3D(x_data, y_data, z_data, cmap, save_path):
    ax = plt.axes(projection='3d')

    ax.scatter3D(z_data, x_data, y_data, c=z_data, cmap=cmap)
    for ii in range(0, 360, 10):
        ax.view_init(elev=0, azim=ii)
        plt.savefig(save_path + "%d.png" % ii)
    plt.show()


def plot_2_channel_3D(x_data_pos, y_data_pos, z_data_pos, cmap_pos, x_data_neg, y_data_neg, z_data_neg, cmap_neg, save_path):
    ax = plt.axes(projection='3d')

    ax.scatter3D(z_data_pos, x_data_pos, y_data_pos, c=z_data_pos, cmap=cmap_pos)
    ax.scatter3D(z_data_neg, x_data_neg, y_data_neg, c=z_data_neg, cmap=cmap_neg)
    for ii in range(0, 360, 10):
        ax.view_init(elev=0, azim=ii)
        plt.savefig(save_path + "%d.png" % ii)
    plt.show()


def show_events(frames, path):
    for ind, frame in enumerate(frames):
        frame = resize_image(frame, 1000)
        cv2.imwrite(path + str(ind) + '.png', frame)
        cv2.imshow("frame", frame)
        cv2.waitKey(200)
    cv2.destroyAllWindows()


def show_partial_matrices(closing_gray_summed_mat, stretched_mat,
                          th_gray_summed_mat, revert_resize, colorized_mask):
    cv2.imshow("closing_gray_summed_mat", closing_gray_summed_mat)
    cv2.imshow("stretched_mat", stretched_mat)
    cv2.imshow("th_gray_summed_mat", th_gray_summed_mat)
    cv2.imshow("revert_resize", revert_resize)
    cv2.imshow("colorized_mask", colorized_mask)
    cv2.waitKey(0)


def generate_gif(save_path, images):
    imgs = []
    for image in images:
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imgs.append(Image.fromarray(color_coverted))

    # duration is the number of milliseconds between frames
    imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)


def plot_voxel_grid(events, sensor_size):
    denoise_transform = tonic.transforms.Denoise(filter_time=10000)
    events_denoised = denoise_transform(events)

    volume = transforms.ToVoxelGrid(sensor_size=sensor_size, n_time_bins=3)(events_denoised)

    fig, axes = plt.subplots(1, len(volume))
    for axis, slice in zip(axes, volume):
        axis.imshow(slice)
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def plot_time_surfaces(events, sensor_size):
    denoise_transform = tonic.transforms.Denoise(filter_time=10000)
    events_denoised = denoise_transform(events)

    surfaces = transforms.ToTimesurface(sensor_size=sensor_size, surface_dimensions=None, tau=10000, decay='exp')(events_denoised)
    n_events = events_denoised.shape[0]
    n_events_per_slice = n_events // 3
    fig, axes = plt.subplots(1, 3)
    for i, axis in enumerate(axes):
        surf = surfaces[(i + 1) * n_events_per_slice - 1]
        axis.imshow(surf[0] - surf[1])
        axis.axis("off")
    plt.tight_layout()

    plt.show()














