from visualize import *
from util import *
from event_frame_generation import *


def create_gray_summed_mat(cropped_frames, len_y, len_x):
    gray_summed_mat = cv2.cvtColor(np.zeros((len_y, len_x, 3), np.uint8), cv2.COLOR_BGR2GRAY)

    for cropped in cropped_frames:
        gray_scaled = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        x_arr, y_arr = np.where(gray_scaled > 0)
        for ind, x in enumerate(x_arr):
            y = y_arr[ind]
            gray_summed_mat[x][y] += 1

    return gray_summed_mat


def generate_average_mask(stretched_mat):
    resized = resize_image(stretched_mat, 2000)
    kernel_size = 11
    closing_gray_summed_mat = cv2.morphologyEx(resized, cv2.MORPH_CLOSE,
                                               make_kernel(kernel_size)[0], iterations=3, borderType=cv2.BORDER_CONSTANT)
    (thresh, th_gray_summed_mat) = cv2.threshold(closing_gray_summed_mat, 100, 255, cv2.THRESH_BINARY)  # 36
    revert_resize = resize_image(th_gray_summed_mat, 5)

    x, y = np.where(revert_resize > 0)
    colorized_mask = cv2.cvtColor(revert_resize, cv2.COLOR_GRAY2BGRA)
    colorized_mask[x, y] = (0, 255, 255, 255)   # YELLOW
    # show_partial_matrices(closing_gray_summed_mat, stretched_mat, th_gray_summed_mat, revert_resize, colorized_mask)

    return colorized_mask


def generate_masks(dataset_entry):
    my_events, target = dataset_entry
    print(target)

    denoise_transform = tonic.transforms.Denoise(filter_time=5000)
    events_denoised = denoise_transform(my_events)

    positive_event_array = generate_event_arrays(events_denoised, 1)
    negative_event_array = generate_event_arrays(events_denoised, 0)

    frames, cropped_frames, len_x, len_y, cropping_positions \
        = generate_fixed_num_events_frames(positive_event_array, negative_event_array)

    gray_summed_mat = create_gray_summed_mat(cropped_frames, len_y, len_x)
    stretched_mat = contrast_stretch(gray_summed_mat, len(cropped_frames))

    colorized_mask = generate_average_mask(stretched_mat)

    label_masked_frames = []
    for ind, frame in enumerate(frames):
        (y0, y1, x0, x1) = cropping_positions[ind]
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        positioned_colorized_mask = np.zeros(result.shape, np.uint8)
        positioned_colorized_mask[y0:y1, x0:x1] = colorized_mask
        result = cv2.addWeighted(result, 1, positioned_colorized_mask, 0.3, 0)
        label_masked_frames.append(result)

    show_events(label_masked_frames, 'frames/label_masked_frame' + str(target) + '_')









