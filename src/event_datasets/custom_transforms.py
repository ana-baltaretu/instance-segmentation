import multiprocessing
import os
import random
import time
from turtle import colormode
import tonic
import tonic.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from multiprocessing import Pool

dt = np.dtype([('x', 'i8'), ('y', 'i8'), ('t', 'i8'), ('p', 'i8')])
TRAIN_PATH = "../data/sorted_NMNIST/train_dataset.npy"
TEST_PATH = "../data/sorted_NMNIST/test_dataset.npy"

TRAIN_MERGED_PATH = "../data/alex_data/train_dataset_merged/"
TEST_MERGED_PATH = "../data/alex_data/test_dataset_merged/"


class Superimposed:
    def __init__(self, background_events, foreground_events):
        self.bkg = np.sort(background_events, order="t")
        self.frg = np.sort(foreground_events, order="t")
        self.merged = None
        self.sensor_size = self.infer_resolution(self.bkg)
        self.bkg_size = self.infer_resolution(self.bkg)
        self.frg_size = self.infer_resolution(self.frg)

    def merge_datasets(self, offset_x=0, offset_y=0):

        # Clean up any noise in the events
        self.remove_background_noise()
        self.remove_foreground_noise()

        # Crop a part of the background
        self.crop_background(centered=True)

        # Set first events to start at timestamp 0
        # self.remove_timestamp_offset_bkg()
        # self.remove_timestamp_offset_frg()

        # Cut background events to end at the same time with foreground
        # self.align_timestamps()

        # Position the digit over the frame
        self.offset_foreground(offset_x=offset_x, offset_y=offset_y)

        # Put the foreground over the background
        self.superimpose()

        # Remove background events covered by foreground
        self.remove_overlapping_events()

        return self.merged

    def infer_resolution(self, events):
        """
        Given events, guess the resolution by looking at the max and min values
        @returns Inferred resolution
        """
        x = events['x'].max() + 1
        y = events['y'].max() + 1
        resolution = (x, y, 2)

        return resolution

    def superimpose(self):
        self.merged = np.concatenate((self.bkg, self.frg))
        self.merged = np.sort(self.merged, order="t")

    def remove_background_noise(self):
        transform = tonic.transforms.Denoise(filter_time=10000)
        self.bkg = transform(self.bkg)

    def remove_foreground_noise(self):
        transform = tonic.transforms.Denoise(filter_time=10000)
        self.frg = transform(self.frg)

    def remove_timestamp_offset_bkg(self):
        transform = tonic.transforms.TimeAlignment()
        self.bkg = transform(self.bkg)

    def remove_timestamp_offset_frg(self):
        transform = tonic.transforms.TimeAlignment()
        self.frg = transform(self.frg)

    def offset_foreground(self, offset_x=0, offset_y=0):
        if offset_x == 0 and offset_y == 0:
            return

        self.frg['x'] += offset_x
        self.frg['y'] += offset_y

        assert np.all(self.frg[:]['x'] <= self.sensor_size[0])
        assert np.all(self.frg[:]['y'] <= self.sensor_size[1])

    def crop_background(self, target_size=(34, 34, 2), centered=False):
        if centered:
            x_min = int(self.bkg_size[0] / 2) - int(target_size[0] / 2)
            y_min = int(self.bkg_size[1] / 2) - int(target_size[1] / 2)

            x_max = x_min + target_size[0]
            y_max = y_min + target_size[1]

            event_mask = (
                (self.bkg["x"] >= x_min)
                * (self.bkg["x"] < x_max)
                * (self.bkg["y"] >= y_min)
                * (self.bkg["y"] < y_max)
            )

            self.bkg = self.bkg[event_mask, ...]
            self.bkg["x"] -= x_min
            self.bkg["y"] -= y_min

            self.sensor_size = target_size
            self.bkg_size = target_size
        else:
            transform = tonic.transforms.RandomCrop(
                sensor_size=self.sensor_size, target_size=target_size)
            self.sensor_size = target_size
            self.bkg = transform(self.bkg)

    def align_timestamps(self):
        if self.frg['t'].max() > self.bkg['t'].max():
            print("WARNING: background events end before foreground events!")

        max_t = self.frg['t'].max()
        self.bkg = self.bkg[self.bkg[:]['t'] <= max_t]

    def remove_overlapping_events(self):
        # use buckets for each timestamp

        # print(self.merged.shape)

        buckets = {}
        buckets2 = {}

        for event in self.merged:
            (x, y, t, p) = event

            if t not in buckets:
                buckets[t] = [event]
            else:
                buckets[t].append(event)

        for t, event_arr in buckets.items():
            buckets2[t] = {}
            for event in event_arr:
                (x, y, t, p) = event

                xy = str(x) + ',' + str(y)

                if xy not in buckets2[t]:
                    buckets2[t][xy] = [event]
                else:
                    buckets2[t][xy].append(event)

        # print([(k, len(v)) if len(v) > 1 else "" for k, v in buckets.items()])
        # print (buckets)

        overlapping = []
        for t, b_xy in buckets2.items():
            for xy, events in b_xy.items():
                if len(events) > 1:
                    overlapping.append(events)

        # print("Events BEFORE removing overlapping:", self.merged.size)
        for overlap in overlapping:
            # if (len(overlap) > 2):
                # print(len(overlap))
            self.merged = self.merged[self.merged != overlap[0]]

        # print("Events AFTER removing overlapping:", self.merged.size)


def to_frames(sensor_size, events, time_bins=20):
    transform = transforms.ToFrame(
        sensor_size=sensor_size, n_time_bins=time_bins)
    return transform(events)


def to_image(sensor_size, events):
    transform = transforms.ToImage(sensor_size)
    return transform(events)


def plot_frames(frames):
    fig, axes = plt.subplots(1, len(frames))
    for axis, frame in zip(axes, frames):
        axis.imshow(frame[1]-frame[0])
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def save_frames(frames, dir_path):
    for idx, frame in enumerate(frames):
        scaled = scale_frame(frame[1]-frame[0], 50)
        plt.imsave(dir_path + '{0}.jpeg'.format(idx), scaled, cmap='gray')


def get_random_entry(ncaltech):
    rand_idx = random.randrange(len(ncaltech))
    entry, label = ncaltech[rand_idx]
    return entry


def perform_merge(ev_foreground, ev_background):
    SI = Superimposed(background_events=np.array(ev_background, dtype=dt), foreground_events=np.array(ev_foreground, dtype=dt))
    return SI.merge_datasets() 

def to_numpy_array(li):
    return np.array(li, dtype=object)

def save_dataset(dataset, path):
    if path == None:
        print("WARNING: Couldn't save dataset, empty path")
        return
    np.save(path, dataset, allow_pickle=True)

def combine_datasets(nmnist=None, ncaltech101=None, id=None, save_path=None):
    assert nmnist is not None
    assert ncaltech101 is not None

    new_dataset = []
    for idx, (ev_digit, label) in enumerate(nmnist):
        # print("Combining: ", idx)
        start_time = time.time()
        merged = perform_merge(ev_digit, ncaltech101)

        new = (merged, label)
        new_dataset.append(new)

        # print("--- %s seconds ---" % (time.time() - start_time))

    new_dataset = to_numpy_array(new_dataset)
    
    if save_path:
        save_dataset(new_dataset, path=save_path)

    print("Done with merge id ", id)
    
    # return new_dataset

def monitor_multiproc(result):
    print("got result: ", len(result))

def split_nmnist_multiproc(nmnist, batches):
    return np.array_split(nmnist, batches)


def combine_datasets_parallel(nmnist, ncaltech101, path):
    assert nmnist is not None
    assert ncaltech101 is not None

    print("Using ", multiprocessing.cpu_count(), " cores")
    cpus = multiprocessing.cpu_count()
    pool = Pool(cpus)



    split_nmnist = split_nmnist_multiproc(nmnist, batches=cpus)
    repeat_ncaltech =  [get_random_entry(ncaltech101) for _ in split_nmnist]
    ids = [idx for idx, _ in enumerate(split_nmnist)]
    paths = [path + str(idx) for idx, _ in enumerate(split_nmnist)]
    args = list(zip(split_nmnist, repeat_ncaltech, ids, paths))


    # start_time = time.time()
    pool.starmap(combine_datasets, args)
    # print("--- Took %s seconds ---" % (time.time() - start_time))

    # final_result = np.concatenate(results)
    # save_dataset(final_result, path)

    
    # print("FINAL results ", len(results))

    # combine_datasets(nmnist=nmnist, ncaltech101=ncaltech101)

def generate_combined_train(train_dataset, ncaltech101_dataset):
    # global_time = time.time()
    # for idx in range(6):
    #     start = idx * 10000
    #     stop = (idx + 1) * 10000
    #     print("Merging batch ids " + str(start) + " to " + str(stop))

        
    #     start_time = time.time()
    #     combine_datasets_parallel(nmnist=train_dataset[start:stop], ncaltech101=ncaltech101_dataset, path="../data/alex_data/train_dataset_merged/batch_" + str(idx) + "_part_")
        
    #     print("DONE with batch ids " + str(start) + " to " + str(stop))
    #     print("--- Took %s seconds ---" % (time.time() - start_time))

    
    # print("--- Whole program took %s seconds ---" % (time.time() - global_time))
    pass


def get_sorted_datasets(train_dataset_path, test_dataset_path):
    # TRAIN
    if os.path.exists(train_dataset_path):
        train_dataset = np.load(train_dataset_path, allow_pickle=True)
    else:
        train_dataset = tonic.datasets.NMNIST(save_to='../data', train=True)

        train_array = [x for _, x in enumerate(train_dataset)]

        # Sort
        train_array.sort(key=lambda tup: tup[1])

        # Save array in file
        if not os.path.exists("../data/sorted_NMNIST"):
            os.mkdir("../data/sorted_NMNIST")

        save_dataset("../data/sorted_NMNIST/train_dataset", to_numpy_array(train_array))

    # TEST
    if os.path.exists(test_dataset_path):
        test_dataset = np.load(test_dataset_path, allow_pickle=True)
    else:
        test_dataset = tonic.datasets.NMNIST(save_to='../data', train=False)
        test_array = [x for _, x in enumerate(test_dataset)]

        # Sort
        test_array.sort(key=lambda tup: tup[1])

        # Save array in file
        if not os.path.exists("../data/sorted_NMNIST"):
            os.mkdir("../data/sorted_NMNIST")
        save_dataset("../data/sorted_NMNIST/test_dataset", to_numpy_array(test_array))

    return train_dataset, test_dataset


def load_dataset_merged(dir_path):
    list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(dir_path, x)) and x.endswith(".npy"),
                        os.listdir(dir_path) ) )

    arrays = None

    for file in list_of_files:
        file_path = os.path.join(dir_path, file)

        a = np.load(file_path, allow_pickle=True)
        # print(a.shape)
        # print(a.dtype)
        if arrays is None:
            arrays = a
        else:
            arrays = np.concatenate((arrays, a))

    # save_dataset(arrays, "")

    return arrays



# Inspiration from Ana Baltaretu:

def save_as_gif(frames, file_name='visualisation.gif'):
    imgs = []
    for frame in frames:
        scaled = np.array(scale_frame(frame[1]-frame[0], 50), dtype=np.int8)
        # color_converted = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        imgs.append(Image.fromarray(scaled))

    # duration is the number of milliseconds between frames
    imgs[0].save(file_name, save_all=True,
                 append_images=imgs[1:], duration=50, loop=0)


def scale_frame(frame, factor):
    """
    img = the image you want to resize
    factor = the factor with which the image is scaled
    """
    width = int(frame.shape[1] * factor)
    height = int(frame.shape[0] * factor)
    size = (width, height)
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return resized
