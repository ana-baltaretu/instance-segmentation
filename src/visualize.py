import matplotlib.pyplot as plt
import tonic
import os
import numpy as np
import cv2
import statistics
import math
import imageio
from PIL import Image
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


def generate_event_arrays(events, polarity, div_rate=100, mult_rate=100):
    """
    Used for splitting the array of events into 3 arrays based on polarity (=sign of the event).

    Input:
    events = stream of events, one entry looks like: (x, y, time, polarity)
    polarity = 1 or 0, 1 for generating array of positive events, 0 for generating array of negative events

    Output (same indices for same event):
    x_data = array with position on the x-axis of the events
    y_data = array with position on the y-axis of the events
    z_data = time of the event (can be flattened with div/mult_rate) -> set both to 1 for no flattening
    """
    filtered_events = list(filter(lambda entry: entry[3] == polarity, events))

    x_data = [entry[0] for entry in filtered_events]
    y_data = [entry[1] for entry in filtered_events]
    z_data = [int(i / div_rate) * mult_rate for i, entry in enumerate(filtered_events)]
    time_data = [int(entry[2]/1000) for entry in filtered_events] # Convert time from microseconds to miliseconds (ms)

    return x_data, y_data, z_data, time_data


def resize_image(img, percentage):
    """
    img = the image you want to resize
    percentage = how big you want it to be
    (ex: 20 for making the image 5 times smaller)
    (ex: 200 for making the image 2 times larger)
    """
    w = int(img.shape[1] * percentage / 100)
    h = int(img.shape[0] * percentage / 100)
    dim = (w, h)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def generate_event_frames(positive_event_array, negative_event_array, window_len=10, img_shape=(34, 34)):
    """
    Takes events in intervals of len=window_len and turns them into a frame, blue=positive, red=negative.
    :param positive_event_array: Positive events (x, y, custom_z, timestamp_in_ms).
    :param negative_event_array: Negative events (x, y, custom_z, timestamp_in_ms).
    :param window_len: Len of time interval that should be allowed in the same frame.
    :param img_shape: How big the output frames should be.
    :return: Array with frames created from event stream.
    """

    img_height, img_width = img_shape
    frames = []
    i, j = 0, 0

    x_data_pos, y_data_pos, z_data_pos, time_data_pos = positive_event_array
    x_data_neg, y_data_neg, z_data_neg, time_data_neg = negative_event_array

    while i < len(time_data_pos) and j < len(time_data_neg):
        current_time = min(time_data_pos[i], time_data_neg[j])
        current_frame = np.zeros((img_height, img_width, 3), np.uint8)
        # current_frame.fill(255)
        while i < len(time_data_pos) and time_data_pos[i] < current_time + window_len:
            x = x_data_pos[i]
            y = y_data_pos[i]
            current_frame[y][x] = (255, 0, 0)   # Blue
            i += 1
        while j < len(time_data_neg) and time_data_neg[j] < current_time + window_len:
            x = x_data_neg[j]
            y = y_data_neg[j]
            current_frame[y][x] = (0, 0, 255)   # Red
            j += 1
        frames.append(current_frame)

    return frames


def generate_fixed_num_events_frames(positive_event_array, negative_event_array, total_frames=20, img_shape=(34, 34)):

    img_height, img_width = img_shape

    frames = []
    i, j = 0, 0

    x_data_pos, y_data_pos, z_data_pos, time_data_pos = positive_event_array
    x_data_neg, y_data_neg, z_data_neg, time_data_neg = negative_event_array

    window_len_pos = int(len(time_data_pos) / total_frames)
    window_len_neg = int(len(time_data_neg) / total_frames)

    len_arr_x, len_arr_y, center_indices = [], [], []

    while i < len(time_data_pos) and j < len(time_data_neg):
        current_i, current_j = i, j
        current_frame = np.zeros((img_height, img_width, 3), np.uint8)
        # current_frame.fill(255)
        while i < len(time_data_pos) and i < current_i + window_len_pos:
            x = x_data_pos[i]
            y = y_data_pos[i]
            current_frame[y][x] = (255, 0, 0)   # Blue
            i += 1
        while j < len(time_data_neg) and j < current_j + window_len_neg:
            x = x_data_neg[j]
            y = y_data_neg[j]
            current_frame[y][x] = (0, 0, 255)   # Red
            j += 1

        # pos_mean_x = statistics.mean(x_data_pos[current_i:i])
        # pos_mean_y = statistics.mean(y_data_pos[current_i:i])
        # neg_mean_x = statistics.mean(x_data_neg[current_j:j])
        # neg_mean_y = statistics.mean(y_data_neg[current_j:j])

        # overall_mean_x = int((pos_mean_x + neg_mean_x) / 2)
        # overall_mean_y = int((pos_mean_y + neg_mean_y) / 2)

        pos_mn_x = min(x_data_pos[current_i:i])
        pos_mn_y = min(y_data_pos[current_i:i])
        pos_mx_x = max(x_data_pos[current_i:i])
        pos_mx_y = max(y_data_pos[current_i:i])

        pos_overall_x = ((pos_mx_x - pos_mn_x) / 2) + pos_mn_x
        pos_overall_y = ((pos_mx_y - pos_mn_y) / 2) + pos_mn_y

        neg_mn_x = min(x_data_neg[current_j:j])
        neg_mn_y = min(y_data_neg[current_j:j])
        neg_mx_x = max(x_data_neg[current_j:j])
        neg_mx_y = max(y_data_neg[current_j:j])

        len_x = max(pos_mx_x, neg_mx_x) - min(pos_mn_x, neg_mn_x)
        len_y = max(pos_mx_y, neg_mx_y) - min(pos_mn_y, neg_mn_y)

        neg_overall_x = ((neg_mx_x - neg_mn_x) / 2) + neg_mn_x
        neg_overall_y = ((neg_mx_y - neg_mn_y) / 2) + neg_mn_y

        overall_x = int((pos_overall_x + neg_overall_x) / 2)
        overall_y = int((pos_overall_y + neg_overall_y) / 2)

        len_arr_x.append(len_x)
        len_arr_y.append(len_y)
        center_indices.append((overall_x, overall_y))

        # Center of the number
        # current_frame[overall_y][overall_x] = (255, 0, 255)

        frames.append(current_frame)

    mean_len_x = statistics.mean(len_arr_x)
    mean_len_y = statistics.mean(len_arr_y)
    hx, hy = math.ceil(mean_len_x/2), math.ceil(mean_len_y/2)

    cropped_frames, cropping_positions = [], []
    for ind, frame in enumerate(frames):
        (cx, cy) = center_indices[ind]
        x0, x1, y0, y1 = cx - hx, cx + hx, cy - hy, cy + hy
        # # Add purple corners
        # print(cx, cy, hx, hy)
        # frame[y0][x0] = (255, 0, 255)
        # frame[y0][x1] = (255, 0, 255)
        # frame[y1][x0] = (255, 0, 255)
        # frame[y1][x1] = (255, 0, 255)

        cropped = frame[y0: y1 + 1, x0: x1 + 1]
        cropped_frames.append(cropped)
        cropping_positions.append((y0, y1 + 1, x0, x1 + 1))

    return frames, cropped_frames, hx * 2 + 1, hy * 2 + 1, cropping_positions


def show_events(frames, path):
    for ind, frame in enumerate(frames):
        frame = resize_image(frame, 1000)
        cv2.imwrite(path + str(ind) + '.png', frame)
        cv2.imshow("frame", frame)
        cv2.waitKey(200)
    cv2.destroyAllWindows()

    # k = cv2.waitKey(0)
    # if k == 27:  # wait for ESC key to exit
    #     cv2.destroyAllWindows()
    # elif k == ord('s'):  # wait for 's' key to save and exit
    #     cv2.imwrite('frames/blank_image.png', blank_image)
    #     cv2.destroyAllWindows()


def generate_gif(save_path, images):
    imgs = []
    for image in images:
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imgs.append(Image.fromarray(color_coverted))

    # duration is the number of milliseconds between frames
    imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)





