from src.my_visualize import *
from src.my_util import *
from src.my_event_frame_generation import *
from mask_matching import get_mask_for_index_and_label


def create_gray_summed_mat(cropped_frames, len_y, len_x):
    gray_summed_mat = cv2.cvtColor(np.zeros((len_y, len_x, 3), np.uint8), cv2.COLOR_BGR2GRAY)

    for cropped in cropped_frames:
        # print(cropped.shape)
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            gray_scaled = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            x_arr, y_arr = np.where(gray_scaled > 0)
            for ind, x in enumerate(x_arr):
                y = y_arr[ind]
                gray_summed_mat[x][y] += 1

    return gray_summed_mat


def color_yellow(image):
    x, y = np.where(image > 0)
    colorized_mask = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    colorized_mask[x, y] = (0, 255, 255, 255)  # YELLOW
    return colorized_mask


def generate_average_mask(stretched_mat):
    resized = resize_image(stretched_mat, 2000)
    kernel_size = 11
    closing_gray_summed_mat = cv2.morphologyEx(resized, cv2.MORPH_CLOSE,
                                               make_kernel(kernel_size)[0], iterations=3, borderType=cv2.BORDER_CONSTANT)
    (thresh, th_gray_summed_mat) = cv2.threshold(closing_gray_summed_mat, 100, 255, cv2.THRESH_BINARY)  # 36
    revert_resize = resize_image(th_gray_summed_mat, 5)
    colorized_mask = color_yellow(revert_resize)

    # show_partial_matrices(closing_gray_summed_mat, stretched_mat, th_gray_summed_mat, revert_resize, colorized_mask)

    return colorized_mask


def generate_masks(dataset_entry, index, last_saved_index, mask_indices_per_label, mnist_dataset):
    my_events, target = dataset_entry
    # plot_frames_denoised(frame_transform, my_events)
    print(target)

    denoise_transform = tonic.transforms.Denoise(filter_time=5000)
    events_denoised = denoise_transform(my_events)
    # events_denoised = my_events

    positive_event_array_denoised = generate_event_arrays(events_denoised, 1)
    negative_event_array_denoised = generate_event_arrays(events_denoised, 0)
    positive_event_array = generate_event_arrays(my_events, 1)
    negative_event_array = generate_event_arrays(my_events, 0)

    # TODO Turn this into generation of fixed window length - DONE

    # frames, cropped_frames, len_x, len_y, cropping_positions, time_frames \
        # = generate_fixed_num_events_frames(positive_event_array, negative_event_array)

    frames, cropped_frames, len_x, len_y, cropping_positions, time_frames \
        = generate_event_frames_with_fixed_time_window(positive_event_array_denoised, negative_event_array_denoised,
                                                       positive_event_array, negative_event_array)

    # Get mask based on index and target

    # TODO: replace this with correct mask - DONEE
    # gray_summed_mat = create_gray_summed_mat(cropped_frames, len_y, len_x)
    # stretched_mat = contrast_stretch(gray_summed_mat, len(cropped_frames))
    # colorized_mask = generate_average_mask(stretched_mat)
    correct_mask = get_mask_for_index_and_label(index - last_saved_index, target,
                                                mnist_dataset, mask_indices_per_label)
    colorized_mask = color_yellow(correct_mask) # TODO Fix this
    # cv2.imshow('correct_mask', correct_mask)
    # cv2.imshow('colorized_mask', colorized_mask)
    # cv2.waitKey(500)
    # print('Colorized', colorized_mask.shape)

    label_masked_frames = []
    colorized_masks = []
    for ind, frame in enumerate(frames):
        # print(np.count_nonzero(frame))
        (y0, y1, x0, x1) = cropping_positions[ind]
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        positioned_colorized_mask = np.zeros(result.shape, np.uint8)
        phh, pww, pcc = positioned_colorized_mask.shape
        hh, ww, cc = colorized_mask.shape
        # print(y1, phh, x1, pww)
        hh -= max(y1 - phh, 0)
        ww -= max(x1 - pww, 0)
        hh = min(hh, phh - y0)
        ww = min(ww, pww - y0)
        # print(hh, ww, y1-y0, x1-x0)
        # print(x0, x0+ww)
        positioned_colorized_mask[y0:(y0+hh), x0:(x0+ww)] = colorized_mask[0:hh, 0:ww]
        result = cv2.addWeighted(result, 1, positioned_colorized_mask, 0.5, 0)
        colorized_masks.append(positioned_colorized_mask)
        label_masked_frames.append(result)

    # show_events(colorized_masks, 'colorized_masks/frame_' + str(target) + '_')
    return frames, colorized_masks, target, time_frames










