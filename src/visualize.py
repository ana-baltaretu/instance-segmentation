import matplotlib.pyplot as plt
import tonic
import os
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

    return x_data, y_data, z_data






