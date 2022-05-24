import numpy as np
import statistics
import math
import cv2

def generate_event_arrays(events, polarity, div_rate=300, mult_rate=100):
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


def gen_extremities(x_data_pos, y_data_pos, x_data_neg, y_data_neg,
                    current_i, i, current_j, j):
    """
    Figures out the edges (min, max extremities) of a frame based on event positions.
    :param x_data_pos:
    :param y_data_pos:
    :param x_data_neg:
    :param y_data_neg:
    :param current_i:
    :param i:
    :param current_j:
    :param j:
    :return:
    """
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

    return len_x, len_y, overall_x, overall_y


def generate_cropped_frames(len_arr_x, len_arr_y, frames, center_indices):
    """
    Creates the cropped frames, their positions and the average centers, as explained in 'generate_fixed_num_events_frames'.
    :param len_arr_x:
    :param len_arr_y:
    :param frames:
    :param center_indices:
    :return:
    """
    mean_len_x = statistics.mean(len_arr_x)
    mean_len_y = statistics.mean(len_arr_y)
    hx, hy = math.ceil(mean_len_x / 2), math.ceil(mean_len_y / 2)

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

    return cropped_frames, hx, hy, cropping_positions


def generate_fixed_num_events_frames(positive_event_array, negative_event_array, total_frames=20, img_shape=(34, 34)):
    """
    Generates event frames with varying amount of events based on the total number of frames.
    Positive and negative events do not interfere with each other's amount for each frame.
    For each frame it also figures out where the center is and returns
    a fixed sized frame based on the average size of an element.
    :param positive_event_array:
    :param negative_event_array:
    :param total_frames:
    :param img_shape:
    :return:
    """
    img_height, img_width = img_shape

    x_data_pos, y_data_pos, z_data_pos, time_data_pos = positive_event_array
    x_data_neg, y_data_neg, z_data_neg, time_data_neg = negative_event_array

    window_len_pos = int(len(time_data_pos) / total_frames)
    window_len_neg = int(len(time_data_neg) / total_frames)

    frames, len_arr_x, len_arr_y, center_indices, time_frames = [], [], [], [], []
    i, j = 0, 0

    while i < len(time_data_pos) and j < len(time_data_neg):
        current_i, current_j = i, j
        current_frame = np.zeros((img_height, img_width, 3), np.uint8)
        time_frame = np.zeros((img_height, img_width, 1), np.uint8)
        # current_frame.fill(255)
        while i < len(time_data_pos) and i < current_i + window_len_pos:
            x = x_data_pos[i]
            y = y_data_pos[i]
            current_frame[y][x] = (255, 0, 0)   # Blue
            time_frame[y][x] = i - current_i    # Latest pixel gets saved
            i += 1
        while j < len(time_data_neg) and j < current_j + window_len_neg:
            x = x_data_neg[j]
            y = y_data_neg[j]
            current_frame[y][x] = (0, 0, 255)   # Red
            time_frame[y][x] = j - current_j    # Latest pixel gets saved
            j += 1

        # print('positive time:', i, current_i)
        # print('negative time:', j, current_j)

        equalized_hist = cv2.equalizeHist(time_frame)

        # create a CLAHE object
        # clahe = cv2.createCLAHE(clipLimit=5.0)
        # clahe_frame = clahe.apply(time_frame)
        #
        # cv2.imshow('time_frame', time_frame)
        # cv2.imshow('equalized', equalized_hist)
        # cv2.imshow('clahe_frame', clahe_frame)
        # cv2.waitKey(200)

        len_x, len_y, overall_x, overall_y = \
            gen_extremities(x_data_pos, y_data_pos, x_data_neg, y_data_neg, current_i, i, current_j, j)

        # Center of the number
        # current_frame[overall_y][overall_x] = (255, 0, 255)

        if np.count_nonzero(current_frame) > 50:
            frames.append(current_frame)
            len_arr_x.append(len_x)
            len_arr_y.append(len_y)
            center_indices.append((overall_x, overall_y))
            time_frames.append(equalized_hist)

    cropped_frames, hx, hy, cropping_positions = \
        generate_cropped_frames(len_arr_x, len_arr_y, frames, center_indices)

    # print(frame.shape)
    # print(hx * 2 + 1, hy * 2 + 1, x1 + 1 - x0, y1 + 1 - y0)
    return frames, cropped_frames, hx * 2 + 1, hy * 2 + 1, cropping_positions, time_frames
















