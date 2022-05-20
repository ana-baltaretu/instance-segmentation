from src.my_label_generation import *


def split_train_test_validation(input_path, output_path, cleanup=False, train_data_percentage=0.8):
    """
    NMNIST is downloaded as a folder with 2 sub-folders ('Train' and 'Test')
    This method splits the 'Train' folder into 'training' and 'testing' folders (which
    will be used for training) and the 'Test' folder becomes the 'validation' set.
    Result is in the N_MNIST folder.
    """

    input_train_path = os.path.join(input_path, 'Train')
    input_test_path = os.path.join(input_path, 'Test')

    assert os.path.exists(input_path)
    assert os.path.exists(input_train_path)
    assert os.path.exists(input_test_path)

    # Reset data folder
    if cleanup is True and os.path.exists(output_path):
        shutil.rmtree(output_path)

    # If it already exists don't generate it again
    if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
        return

    # Make directory if it doesn't exist
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    # Copy 'Test' to 'validation'
    print(input_test_path)
    print(os.path.join(output_path, 'validation'))
    shutil.copytree(input_test_path, os.path.join(output_path, 'validation'))
    print("Copied!")

    generate_directories(output_path, ['training', 'testing'])

    for number_folder in os.listdir(input_train_path):
        input_number_path = os.path.join(input_train_path, number_folder)

        training_path = os.path.join(os.path.join(output_path, 'training'), number_folder)
        if os.path.exists(training_path) is False:
            os.mkdir(training_path)

        testing_path = os.path.join(os.path.join(output_path, 'testing'), number_folder)
        if os.path.exists(testing_path) is False:
            os.mkdir(testing_path)

        print(input_number_path)
        print(training_path)
        print(testing_path)
        print()

        for file in os.listdir(input_number_path):
            rand = random.random()
            current_file_path = os.path.join(input_number_path, file)
            if rand < train_data_percentage:
                shutil.copyfile(current_file_path, os.path.join(training_path, file))
                # print('training')
            else:
                shutil.copyfile(current_file_path, os.path.join(testing_path, file))
                # print('testing')


def save_images(chosen_directory, dataset, skip, train_data_percentage=1, secondary_chosen_directory=''):
    for i, entry in enumerate(dataset):
        if i % skip == 0:
            frames, colorized_masks, target = generate_masks(entry)

            if random.random() < train_data_percentage:
                target_path = os.path.join(chosen_directory, str(target))
            else:
                target_path = os.path.join(secondary_chosen_directory, str(target))

            print(target_path)

            if os.path.exists(target_path) is False:
                os.mkdir(target_path)
            for j, frame in enumerate(frames):
                cv2.imwrite(target_path + '/frame' + str(i) + '_' + str(j) + '.png', frame)
                cv2.imwrite(target_path + '/mask' + str(i) + '_' + str(j) + '.png', colorized_masks[j])


def generate_rgbd_images_and_masks(train_dataset, test_dataset, output_path, cleanup=False, skip=1000):
    """
    Converting the input binary images to RGB-D images and create their masks.
    """

    cleanup_files(output_path, cleanup)

    # If it already exists don't generate it again
    if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
        return

    generate_directories(output_path, ['training', 'testing', 'validation'])
    training_path = os.path.join(output_path, 'training')
    testing_path = os.path.join(output_path, 'testing')
    validation_path = os.path.join(output_path, 'validation')

    print('--------------------------- Validation ---------------------------')
    save_images(validation_path, test_dataset, skip)
    print('--------------------------- Train&Test ---------------------------')
    save_images(training_path, train_dataset, skip, train_data_percentage=0.8, secondary_chosen_directory=testing_path)