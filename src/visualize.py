import matplotlib.pyplot as plt
import tonic
import os
import numpy as np
import cv2
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


def generate_event_frames(positive_event_array, negative_event_array, window_len=20, img_shape=(34, 34)):
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
        while i < len(time_data_pos) and time_data_pos[i] < current_time + window_len:
            x = x_data_pos[i]
            y = y_data_pos[i]
            current_frame[x][y] = (255, 0, 0)   # Blue
            i += 1
        while j < len(time_data_neg) and time_data_neg[j] < current_time + window_len:
            x = x_data_neg[j]
            y = y_data_neg[j]
            current_frame[x][y] = (0, 0, 255)   # Red
            j += 1
        frames.append(current_frame)

    return frames


def show_events(frames):
    for ind, frame in enumerate(frames):
        frame = resize_image(frame, 1000)
        cv2.imwrite('frames/frame' + str(ind) + '.png', frame)
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    # k = cv2.waitKey(0)
    # if k == 27:  # wait for ESC key to exit
    #     cv2.destroyAllWindows()
    # elif k == ord('s'):  # wait for 's' key to save and exit
    #     cv2.imwrite('frames/blank_image.png', blank_image)
    #     cv2.destroyAllWindows()








