import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
from skimage.color import rgb2gray
import cv2
import os
import random
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

def gen_image_paths_train(training_image_path_list, num_level):
    all_images_image_paths = []
    all_images_image_labels = []

    for i in training_image_path_list:

        slide_path = i

        img_num = slide_path.split('_')[1].strip(".tif")

        data_root_tumor = pathlib.Path('data/' + img_num + '/level_' + str(num_level) + '/tumor')
        all_image_paths_tumor = list(data_root_tumor.glob('*'))
        num_tumor_images = len(all_image_paths_tumor)

        data_root_notumor = pathlib.Path('data/' + img_num + '/level_' + str(num_level) + '/no_tumor')
        all_image_paths_notumor = list(data_root_notumor.glob('*'))
        random.shuffle(all_image_paths_notumor)
        all_image_paths_notumor = all_image_paths_notumor[0:num_tumor_images]

        all_image_paths = [str(path) for path in all_image_paths_tumor + all_image_paths_notumor]
        random.shuffle(all_image_paths)

        data_root = pathlib.Path('data/' + img_num + '/level_' + str(num_level))
        label_names = sorted(item.name for item in data_root.glob('*') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))

        all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in all_image_paths]

        #update all image path lists
        all_images_image_paths = all_images_image_paths + all_image_paths
        all_images_image_labels = all_images_image_labels + all_image_labels

    return all_images_image_paths, all_images_image_labels

# Load and preprocess image.
def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)

#Preprocess image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [128, 128])
    image /= 255.0  # normalize to [0,1] range

    return image

# Generate image paths
def gen_image_paths(slide_path, level_num):
    img_num = slide_path.split('_')[1].strip(".tif")
    img_test_folder = 'tissue_only'

    data_root = pathlib.Path('data/' + img_num + '/level_' + str(level_num) +'/' + img_test_folder)

    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    return  all_image_paths

# Tumor predict mask.
def tumor_predict_mask(test, all_image_paths, depth, width):

    test = test[0:len(all_image_paths), :]
    img_num = np.zeros(len(all_image_paths))
    for i in range(len(all_image_paths)):
        img_num[i] = int(all_image_paths[i].strip('.jpg').split('/')[-1].split('_')[-1])

    # depth, width = int(np.ceil(slide_image.shape[0] / pixel_num)), int(np.ceil(slide_image.shape[1] / pixel_num))

    predictions = np.zeros((depth, width))
    conf_threshold = 0.85

    for i in range(len(test)):
        y = int(img_num[i] // width)
        x = int(np.mod(img_num[i], width))
        predictions[y, x] = int(test[i][1] > conf_threshold)

    return predictions

def create_tf_dataset_train(all_image_paths_1, all_image_paths_2, all_image_paths_3, all_image_labels):

    path_ds_1 = tf.data.Dataset.from_tensor_slices(all_image_paths_1)
    image_ds_1 = path_ds_1.map(load_and_preprocess_image, num_parallel_calls=8)

    path_ds_2 = tf.data.Dataset.from_tensor_slices(all_image_paths_2)
    image_ds_2 = path_ds_2.map(load_and_preprocess_image, num_parallel_calls=8)

    path_ds_3 = tf.data.Dataset.from_tensor_slices(all_image_paths_3)
    image_ds3_3 = path_ds_3.map(load_and_preprocess_image, num_parallel_calls=8)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

    image_label_ds = tf.data.Dataset.zip(((image_ds_1,image_ds_2, image_ds3_3), label_ds))

    BATCH_SIZE = 4

    steps_per_epoch = int(np.ceil(len(all_image_paths_1)/BATCH_SIZE))

    # Setting a shuffle buffer size larger than the dataset ensures that the data is completely shuffled.
    ds = image_label_ds.repeat()
    ds = ds.shuffle(buffer_size=4000)
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetches batches, asynchronously while the model is training.
    ds = ds.prefetch(1)


    return ds, steps_per_epoch

# Create tf dataset
def create_tf_dataset(all_image_paths_1, all_image_paths_2, all_image_paths_3):
    path_ds_1 = tf.data.Dataset.from_tensor_slices(all_image_paths_1)
    image_ds_1 = path_ds_1.map(load_and_preprocess_image, num_parallel_calls=8)

    path_ds_2 = tf.data.Dataset.from_tensor_slices(all_image_paths_2)
    image_ds_2 = path_ds_2.map(load_and_preprocess_image, num_parallel_calls=8)

    path_ds_3 = tf.data.Dataset.from_tensor_slices(all_image_paths_3)
    image_ds_3 = path_ds_3.map(load_and_preprocess_image, num_parallel_calls=8)
    image_test_ds = tf.data.Dataset.zip(((image_ds_1,image_ds_2, image_ds_3),))

    ## Dataset parameters
    BATCH_SIZE = 4

    steps_per_epoch = int(np.ceil(len(all_image_paths_1) / BATCH_SIZE))
    # Setting a shuffle buffer size larger than the dataset ensures that the data is completely shuffled.
    ds = image_test_ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetches batches, asynchronously while the model is training.
    ds = ds.prefetch(1)

    return ds, steps_per_epoch

# Test
def test_part(training_image_path, model, tissue_regions, slide_image_test,
                mask_image, depth, width, num_level_1, num_level_2, num_level_3):

    ## Generate image paths and labels
    all_image_paths_1 = gen_image_paths(training_image_path, num_level_1)

    # create the second file path to mimic the 1st
    all_image_paths_2 = []
    for i in all_image_paths_1:
        split_str = i.split('/')
        split_str[2] = 'level_'+str(num_level_2)
        path_2_string = ''
        for j in split_str:
            if j == split_str[-1]:
                path_2_string = path_2_string + j
            else:
                path_2_string = path_2_string + j + '/'
        all_image_paths_2.append(path_2_string)

    all_image_paths_3 = []
    for i in all_image_paths_1:
        split_str = i.split('/')
        split_str[2] = 'level_'+str(num_level_3)
        path_3_string = ''
        for j in split_str:
            if j == split_str[-1]:
                path_3_string = path_3_string + j
            else:
                path_3_string = path_3_string + j + '/'
        all_image_paths_3.append(path_3_string)


    ## Create tf.Dataset for testing
    ds_test, steps_per_epoch_test = create_tf_dataset(all_image_paths_1, all_image_paths_2, all_image_paths_3)

    ## Predict on test data
    test_predicts = model.predict(ds_test, steps = steps_per_epoch_test)

    ## Create mask containing test predictions
    predictions = tumor_predict_mask(test_predicts, all_image_paths_1, depth, width)

    print('Slide image')
    fig1, ax1 = plt.subplots()
    plt.axis('off')
    plt.imshow(slide_image_test)

    print('Predicted image')
    fig2, ax2 = plt.subplots()
    plt.axis('off')
    plt.imshow(predictions)

    print('Actual mask')
    fig3, ax3 = plt.subplots()
    plt.axis('off')
    plt.imshow(mask_image)

    heatmap_evaluation(predictions, mask_image, tissue_regions)


def train_part(training_image_path_list, num_level_1, num_level_2, num_level_3 ):

    # change input here from a specific image to an image path
    all_image_paths_1, all_image_labels_1 = gen_image_paths_train(training_image_path_list, num_level_1)

    # create the second file path to mimic the 1st
    all_image_paths_2 = []
    for i in all_image_paths_1:
        split_str = i.split('/')
        split_str[2] = 'level_'+str(num_level_2)
        path_2_string = ''
        for j in split_str:
            if j == split_str[-1]:
                path_2_string = path_2_string + j
            else:
                path_2_string = path_2_string + j + '/'
        all_image_paths_2.append(path_2_string)

    all_image_paths_3 = []
    for i in all_image_paths_1:
        split_str = i.split('/')
        split_str[2] = 'level_'+str(num_level_3)
        path_3_string = ''
        for j in split_str:
            if j == split_str[-1]:
                path_3_string = path_3_string + j
            else:
                path_3_string = path_3_string + j + '/'
        all_image_paths_3.append(path_3_string)

    ## Create tf.Dataset for training
    ds, steps_per_epoch = create_tf_dataset_train(all_image_paths_1, all_image_paths_2, all_image_paths_3, all_image_labels_1)

    return ds, steps_per_epoch

 # Heatmap evaluation
def heatmap_evaluation(predictions, mask_image, tissue_regions):

    # we only need to evaluate on areas which are tissue.
    # correct non tumor prediction count would be higher if we get credit for
    # predicting gray areas aren't tumors

    # find out the correct amount to scale predictions to match image
    scale = int(mask_image.shape[0]/predictions.shape[0])

    # create scaled prediction matix
    predictions_scaled = np.kron(predictions, np.ones((scale, scale)))

    # reshape everything to a 1D vector for easy computation
    predictions_scaled = predictions_scaled.reshape(-1)
    mask_image = mask_image.reshape(-1)
    tissue_regions = tissue_regions.reshape(-1)

    # only include entries that have tissue
    predictions_scaled = predictions_scaled[tissue_regions == 1]
    mask_image = mask_image[tissue_regions == 1]

    # evaluate
    r = roc_auc_score(mask_image, predictions_scaled)
    print("ROC AUC score: ", r)

    print(classification_report(mask_image, predictions_scaled))

    # Creates a confusion matrix
    cm = confusion_matrix(mask_image, predictions_scaled)

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm,
                         index = ['no tumor', 'tumor'],
                         columns = ['no tumor', 'tumor'])

    plt.figure(figsize=(5.5,4))
    sns.heatmap(cm_df, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
