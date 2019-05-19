import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
from skimage.color import rgb2gray
import cv2
import os
import random
import pathlib
import openslide
import PIL
from PIL import Image
from openslide.lowlevel import *
from openslide.lowlevel import _convert


def _load_image_lessthan_2_29(buf, size):
    '''buf must be a mutable buffer.'''
    _convert.argb2rgba(buf)
    return PIL.Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)


def _load_image_morethan_2_29(buf, size):
    '''buf must be a buffer.'''

    # Load entire buffer at once if possible
    MAX_PIXELS_PER_LOAD = (1 << 29) - 1
    # Otherwise, use chunks smaller than the maximum to reduce memory
    # requirements
    PIXELS_PER_LOAD = 1 << 26

    def do_load(buf, size):
        '''buf can be a string, but should be a ctypes buffer to avoid an
        extra copy in the caller.'''
        # First reorder the bytes in a pixel from native-endian aRGB to
        # big-endian RGBa to work around limitations in RGBa loader
        rawmode = (sys.byteorder == 'little') and 'BGRA' or 'ARGB'
        buf = PIL.Image.frombuffer('RGBA', size, buf, 'raw', rawmode, 0, 1)
        # Image.tobytes() is named tostring() in Pillow 1.x and PIL
        buf = (getattr(buf, 'tobytes', None) or buf.tostring)()
        # Now load the image as RGBA, undoing premultiplication
        return PIL.Image.frombuffer('RGBA', size, buf, 'raw', 'RGBa', 0, 1)

    # Fast path for small buffers
    w, h = size
    if w * h <= MAX_PIXELS_PER_LOAD:
        return do_load(buf, size)

    # Load in chunks to avoid OverflowError in PIL.Image.frombuffer()
    # https://github.com/python-pillow/Pillow/issues/1475
    if w > PIXELS_PER_LOAD:
        # We could support this, but it seems like overkill
        raise ValueError('Width %d is too large (maximum %d)' %
                         (w, PIXELS_PER_LOAD))
    rows_per_load = PIXELS_PER_LOAD // w
    img = PIL.Image.new('RGBA', (w, h))
    for y in range(0, h, rows_per_load):
        rows = min(h - y, rows_per_load)
        if sys.version[0] == '2':
            chunk = buffer(buf, 4 * y * w, 4 * rows * w)
        else:
            # PIL.Image.frombuffer() won't take a memoryview or
            # bytearray, so we can't avoid copying
            chunk = memoryview(buf)[y * w:(y + rows) * w].tobytes()
        img.paste(do_load(chunk, (w, rows)), (0, y))
    return img


# The function below is used to load the images and tumor mask images.
def load_image(slide_path, tumor_mask_path):

    # Loading the slide image and the tumor mask image
    slide = open_slide(slide_path)
    tumor_mask = open_slide(tumor_mask_path)

    # Checking if the dimensions of the mask image and the slide image match or not
    for i in range(len(slide.level_dimensions)-1):
        assert tumor_mask.level_dimensions[i][0] == slide.level_dimensions[i][0]
        assert tumor_mask.level_dimensions[i][1] == slide.level_dimensions[i][1]

    # Verify downsampling works as expected
    width, height = slide.level_dimensions[7]
    assert width * slide.level_downsamples[7] == slide.level_dimensions[0][0]
    assert height * slide.level_downsamples[7] == slide.level_dimensions[0][1]

    return slide, tumor_mask


def read_slide(slide, x, y, level, width, height, as_float=False):

    # Reading the slides and converting them into a RGB numpy array
    openslide.lowlevel._load_image = _load_image_morethan_2_29
    im = slide.read_region((x, y), level, (width, height))
    im = im.convert('RGB')  # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im


def find_tissue_pixels(image, intensity=0.8):

    # Finding the pixels having value less than or equal to the intensity value
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    return zip(indices[0], indices[1])


def apply_mask(im, mask, color=1):

    # Applies the mask to the slides image
    masked = np.zeros((im.shape[0], im.shape[1]))
    for x, y in mask:
        masked[x][y] = color
    return masked


# This function makes the directories required during testing
def initialize_directories_test(slide_path, level):

    BASE_DIR = os.getcwd()

    img_num = slide_path.split('_')[1].strip(".tif")

    DATA = 'data/'
    IMG_NUM_FOLDER = img_num + '/'
    LEVEL_FOLDER = 'level_'+str(level)+'/'
    TISSUE_FOLDER = 'tissue_only/'
    ALL_FOLDER = 'all/'

    DATA_DIR = os.path.join(BASE_DIR, DATA)
    IMG_NUM_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER)
    LEVEL_NUM_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER)
    TISSUE_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER, TISSUE_FOLDER)
    ALL_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER, ALL_FOLDER)

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(IMG_NUM_DIR):
        os.mkdir(IMG_NUM_DIR)
    if not os.path.exists(LEVEL_NUM_DIR):
        os.mkdir(LEVEL_NUM_DIR)
    if not os.path.exists(TISSUE_DIR):
        os.mkdir(TISSUE_DIR)
    if not os.path.exists(ALL_DIR):
        os.mkdir(ALL_DIR)

    return DATA + IMG_NUM_FOLDER + LEVEL_FOLDER + TISSUE_FOLDER, DATA + IMG_NUM_FOLDER + LEVEL_FOLDER + ALL_FOLDER


# This function is used to create the directories during training
def initialize_directories(slide_path, level):
    BASE_DIR = os.getcwd()

    img_num = slide_path.split('_')[1].strip(".tif")

    DATA = 'data/'
    IMG_NUM_FOLDER = img_num + '/'
    LEVEL_FOLDER = 'level_'+str(level)+'/'
    TUMOR_FOLDER = 'tumor/'
    NO_TUMOR_FOLDER = 'no_tumor/'

    DATA_DIR = os.path.join(BASE_DIR, DATA)
    IMG_NUM_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER)
    LEVEL_NUM_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER)
    TUMOR_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER, TUMOR_FOLDER)
    NO_TUMOR_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER, NO_TUMOR_FOLDER)

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(IMG_NUM_DIR):
        os.mkdir(IMG_NUM_DIR)
    if not os.path.exists(LEVEL_NUM_DIR):
        os.mkdir(LEVEL_NUM_DIR)
    if not os.path.exists(TUMOR_DIR):
        os.mkdir(TUMOR_DIR)
    if not os.path.exists(NO_TUMOR_DIR):
        os.mkdir(NO_TUMOR_DIR)

    return DATA + IMG_NUM_FOLDER + LEVEL_FOLDER + TUMOR_FOLDER, DATA + IMG_NUM_FOLDER + LEVEL_FOLDER + NO_TUMOR_FOLDER


# Save the image as tissue or not
def split_image_test(im, tissue_mask, num_pixels, level_num, slide_path):
    x, y = im.shape[0], im.shape[1]
    x_count, y_count = int(np.ceil(x / num_pixels)), int(np.ceil(y / num_pixels))

    tissue_folder, all_folder = initialize_directories_test(slide_path, level_num)

    try:
        for i in range(x_count):
            for j in range(y_count):
                im_slice = np.zeros((num_pixels, num_pixels, 3))
                im_tissue_slice = np.zeros((num_pixels, num_pixels, 3))
                tissue_mask_slice = np.zeros((num_pixels, num_pixels))

                string_name = 'img_' + str(i * y_count + j)

                # Logic to handle the edges of the images
                if i == x_count - 1:
                    ub_x = x
                    assign_x = x - (x_count - 1) * num_pixels
                else:
                    ub_x = (i + 1) * num_pixels
                    assign_x = num_pixels

                if j == y_count - 1:
                    ub_y = y
                    assign_y = y - (y_count - 1) * num_pixels
                else:
                    ub_y = (j + 1) * num_pixels
                    assign_y = num_pixels

                tissue_mask_slice[0:assign_x, 0:assign_y] = tissue_mask[(i * num_pixels):ub_x, (j * num_pixels):ub_y]

                try:
                    if np.mean(tissue_mask_slice) > 0.7:
                        im_tissue_slice[0:assign_x, 0:assign_y, :] = im[(i * num_pixels):ub_x, (j * num_pixels):ub_y, :]
                        im_file_name_tissue = tissue_folder + string_name + ".jpg"
                        cv2.imwrite(im_file_name_tissue, im_tissue_slice)

                    im_slice[0:assign_x, 0:assign_y, :] = im[(i * num_pixels):ub_x, (j * num_pixels):ub_y, :]
                    im_file_name_all = all_folder + string_name + ".jpg"
                    cv2.imwrite(im_file_name_all, im_slice)

                except Exception as oerr:
                    print('Error with saving:', oerr)

    except Exception as oerr:
        print('Error with slicing:', oerr)


# The function where the images are stored into tumor or no tumor folders
def split_image_and_mask(im, tumor_mask, tissue_mask,  num_pixels, level, slide_path):

    x, y = im.shape[0], im.shape[1]

    # Find the number of image slices that the original image will split into
    x_count, y_count = int(np.ceil(x / num_pixels)), int(np.ceil(y/num_pixels))

    tumor_folder, no_tumor_folder = initialize_directories(slide_path, level)

    try:
        for i in range(x_count):
            for j in range(y_count):
                im_slice = np.zeros((num_pixels, num_pixels, 3))
                tissue_mask_slice = np.zeros((num_pixels, num_pixels))
                tumor_mask_slice = np.zeros((num_pixels, num_pixels))

                string_name = 'img_' + str(i * y_count + j)

                # Logic to handle end conditions
                if i == x_count-1:
                    ub_x = x
                    assign_x = x - (x_count-1)*num_pixels
                else:
                    ub_x = (i+1) * num_pixels
                    assign_x = num_pixels

                if j == y_count-1:
                    ub_y = y
                    assign_y = y - (y_count-1)*num_pixels
                else:
                    ub_y = (j+1) * num_pixels
                    assign_y = num_pixels

                # Assign the pixels to the slice of the tissue mas
                tissue_mask_slice[0:assign_x, 0:assign_y] = tissue_mask[(i*num_pixels) :ub_x, (j * num_pixels) :ub_y]

                try:
                    if np.mean(tissue_mask_slice) > 0.7:
                        im_slice[0:assign_x, 0:assign_y, :] = im[(i*num_pixels) :ub_x, (j * num_pixels) :ub_y, :]
                        tumor_mask_slice[0:assign_x, 0:assign_y] = tumor_mask[(i*num_pixels) :ub_x, (j * num_pixels) :ub_y]

                        if np.max(tumor_mask_slice) > 0:
                            im_file_name = tumor_folder + string_name + ".jpg"
                        else:
                            im_file_name = no_tumor_folder + string_name + ".jpg"

                        cv2.imwrite(im_file_name, im_slice)

                except Exception as oerr:
                    print('Error with saving:', oerr)

    except Exception as oerr:
        print('Error with slicing:', oerr)


def load_second_level(slide_path, input_level, num_input_pixels, output_level, num_output_pixels):
    img_num = slide_path.split('_')[1].strip(".tif")

    BASE_DIR = os.getcwd()
    DATA = 'data/'
    LEVEL_INPUT_FOLDER = 'level_' + str(input_level) + '/'
    LEVEL_OUTPUT_FOLDER = 'level_' + str(output_level) + '/'
    TISSUE_FOLDER = 'tissue_only/'
    ALL_FOLDER = 'all/'

    TISSUE_DIR_INPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_INPUT_FOLDER, TISSUE_FOLDER)
    ALL_DIR_INPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_INPUT_FOLDER, ALL_FOLDER)
    LEVEL_DIR_OUTPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_OUTPUT_FOLDER)
    TISSUE_DIR_OUTPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_OUTPUT_FOLDER, TISSUE_FOLDER)
    ALL_DIR_OUTPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_OUTPUT_FOLDER, ALL_FOLDER)

    if not os.path.exists(LEVEL_DIR_OUTPUT):
        os.mkdir(LEVEL_DIR_OUTPUT)
    if not os.path.exists(TISSUE_DIR_OUTPUT):
        os.mkdir(TISSUE_DIR_OUTPUT)
    if not os.path.exists(ALL_DIR_OUTPUT):
        os.mkdir(ALL_DIR_OUTPUT)

    data_root_tissue_input = pathlib.Path(TISSUE_DIR_INPUT)
    all_image_paths_tissue_input = list(data_root_tissue_input.glob('*'))
    all_paths_tissue_str_input = [str(path) for path in all_image_paths_tissue_input]
    num_tissue_images_input = len(all_image_paths_tissue_input)

    data_root_all_input = pathlib.Path(ALL_DIR_INPUT)
    all_image_paths_all_input = list(data_root_all_input.glob('*'))
    all_paths_all_str_input = [str(path) for path in all_image_paths_all_input]
    num_all_images_input = len(all_paths_all_str_input)

    slide = open_slide(slide_path)
    input_width, input_height = slide.level_dimensions[input_level][0], slide.level_dimensions[input_level][1]
    output_width, output_height = slide.level_dimensions[output_level][0], slide.level_dimensions[output_level][1]
    slide = read_slide(slide,
                       x=0,
                       y=0,
                       level=output_level,
                       width=output_width,
                       height=output_height)

    # Find number of images that can fit in x and y direction of input given input number of pixels
    row_count, col_count = int(np.ceil(input_height / num_input_pixels)), int(np.ceil(input_width / num_input_pixels))

    for i in all_paths_tissue_str_input + all_paths_all_str_input:
        try:

            img_index = i.split('_')[-1].strip(".jpg")

            start_input_row_count, start_input_col_count = (int(img_index) // col_count), (int(img_index) % col_count)
            start_input_row_index = start_input_row_count * num_input_pixels
            start_input_col_index = start_input_col_count * num_input_pixels

            scale_factor = 2 ** (output_level - input_level)
            shift_middle = (1 - 1/scale_factor)/2

            start_output_row = start_input_row_index * 1/scale_factor
            start_output_col = start_input_col_index * 1/scale_factor

            shift_pixels_up_left = shift_middle * num_output_pixels
            shift_pixels_down_right = (1 - shift_middle) * num_output_pixels

            temp_shift_row_top = start_output_row - shift_pixels_up_left
            temp_shift_col_left = start_output_col - shift_pixels_up_left
            temp_shift_row_bottom = start_output_row + shift_pixels_down_right
            temp_shift_col_right = start_output_col + shift_pixels_down_right

            start_slice_top = 0
            start_slice_left = 0
            end_slice_bottom = num_output_pixels
            end_slice_right = num_output_pixels

            start_image_top = int(np.max((temp_shift_row_top, 0)))
            start_image_left = int(np.max((temp_shift_col_left, 0)))
            end_image_bottom = int(np.min((temp_shift_row_bottom, output_height)))
            end_image_right = int(np.min((temp_shift_col_right,  output_width)))

            if temp_shift_row_top < 0:
                start_slice_top = int(-temp_shift_row_top)

            if temp_shift_col_left < 0:
                start_slice_left = int(-temp_shift_col_left)

            if temp_shift_row_bottom > output_height:
                end_slice_bottom = int(num_output_pixels - (temp_shift_row_bottom - output_height))

            if temp_shift_col_right > output_width:
                end_slice_right = int(num_output_pixels - (temp_shift_col_right - output_width))

            output_slice = np.zeros((num_output_pixels, num_output_pixels, 3))
            output_slice[start_slice_top: end_slice_bottom, start_slice_left: end_slice_right] = \
            slide[start_image_top: end_image_bottom, start_image_left: end_image_right]

            if i in all_paths_tissue_str_input:
                save_path = TISSUE_DIR_OUTPUT
            else:
                save_path = ALL_DIR_OUTPUT

            output_file_name = save_path + 'img_' + str(img_index) + '.jpg'

            try:
                cv2.imwrite(output_file_name, output_slice)
            except Exception as oerr:
                print('Error with saving:', oerr)

        except Exception as oerr:
            print('Error with slice:', oerr)

    data_root_tissue_output = pathlib.Path(TISSUE_DIR_OUTPUT)
    all_image_paths_tissue_output = list(data_root_tissue_output.glob('*'))
    all_paths_tissue_str_output = [str(path) for path in all_image_paths_tissue_output]
    num_tissue_images_output = len(all_paths_tissue_str_output)

    data_root_all_output = pathlib.Path(ALL_DIR_OUTPUT)
    all_image_paths_all_output = list(data_root_all_output.glob('*'))
    all_paths_all_str_output = [str(path) for path in all_image_paths_all_output]
    num_all_images_output = len(all_paths_all_str_output)

    if (num_tissue_images_output != num_tissue_images_input) or (num_all_images_output != num_all_images_input):
        print('ERROR: Number of output images not the same as number of input images')


def save_second_level(slide_path_list, input_level, num_input_pixels, output_level, num_output_pixels):
    for slide_path in slide_path_list:
        img_num = slide_path.split('_')[1].strip(".tif")

        BASE_DIR = os.getcwd()
        DATA = 'data/'
        LEVEL_INPUT_FOLDER = 'level_' + str(input_level) + '/'
        LEVEL_OUTPUT_FOLDER = 'level_' + str(output_level) + '/'
        TUMOR_FOLDER = 'tumor/'
        NO_TUMOR_FOLDER = 'no_tumor/'

        TUMOR_DIR_INPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_INPUT_FOLDER, TUMOR_FOLDER)
        NO_TUMOR_DIR_INPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_INPUT_FOLDER, NO_TUMOR_FOLDER)
        LEVEL_DIR_OUTPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_OUTPUT_FOLDER)
        TUMOR_DIR_OUTPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_OUTPUT_FOLDER, TUMOR_FOLDER)
        NO_TUMOR_DIR_OUTPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_OUTPUT_FOLDER, NO_TUMOR_FOLDER)

        if not os.path.exists(LEVEL_DIR_OUTPUT):
            os.mkdir(LEVEL_DIR_OUTPUT)
        if not os.path.exists(TUMOR_DIR_OUTPUT):
            os.mkdir(TUMOR_DIR_OUTPUT)
        if not os.path.exists(NO_TUMOR_DIR_OUTPUT):
            os.mkdir(NO_TUMOR_DIR_OUTPUT)

        data_root_tumor_input = pathlib.Path(TUMOR_DIR_INPUT)
        all_image_paths_tumor_input = list(data_root_tumor_input.glob('*'))
        all_paths_tumor_str_input = [str(path) for path in all_image_paths_tumor_input]
        num_tumor_images_input = len(all_image_paths_tumor_input)

        data_root_notumor_input = pathlib.Path(NO_TUMOR_DIR_INPUT)
        all_image_paths_notumor_input = list(data_root_notumor_input.glob('*'))
        all_paths_notumor_str_input = [str(path) for path in all_image_paths_notumor_input]
        num_notumor_images_input = len(all_paths_notumor_str_input)

        slide = open_slide(slide_path)
        input_width, input_height = slide.level_dimensions[input_level][0], slide.level_dimensions[input_level][1]
        output_width, output_height = slide.level_dimensions[output_level][0], slide.level_dimensions[output_level][1]

        slide = read_slide(slide,
                           x=0,
                           y=0,
                           level=output_level,
                           width=output_width,
                           height=output_height)

        # Find number of images that can fit in x and y direction of input given input number of pixels
        row_count, col_count = int(np.ceil(input_height / num_input_pixels)), int(np.ceil(input_width / num_input_pixels))

        for i in all_paths_tumor_str_input + all_paths_notumor_str_input:
            try:

                img_index = i.split('_')[-1].strip(".jpg")

                start_input_row_count, start_input_col_count = (int(img_index) // col_count), (int(img_index) % col_count)
                start_input_row_index = start_input_row_count * num_input_pixels
                start_input_col_index = start_input_col_count * num_input_pixels

                scale_factor = 2 ** (output_level - input_level)
                shift_middle = (1 - 1/scale_factor)/2

                start_output_row = start_input_row_index * 1/scale_factor
                start_output_col = start_input_col_index * 1/scale_factor

                shift_pixels_up_left = shift_middle * num_output_pixels
                shift_pixels_down_right = (1 - shift_middle) * num_output_pixels

                temp_shift_row_top = start_output_row - shift_pixels_up_left
                temp_shift_col_left = start_output_col - shift_pixels_up_left
                temp_shift_row_bottom = start_output_row + shift_pixels_down_right
                temp_shift_col_right = start_output_col + shift_pixels_down_right

                start_slice_top = 0
                start_slice_left = 0
                end_slice_bottom = num_output_pixels
                end_slice_right = num_output_pixels

                start_image_top = int(np.max((temp_shift_row_top, 0)))
                start_image_left = int(np.max((temp_shift_col_left, 0)))
                end_image_bottom = int(np.min((temp_shift_row_bottom, output_height)))
                end_image_right = int(np.min((temp_shift_col_right,  output_width)))

                if temp_shift_row_top < 0:
                    start_slice_top = int(-temp_shift_row_top)

                if temp_shift_col_left < 0:
                    start_slice_left = int(-temp_shift_col_left)

                if temp_shift_row_bottom > output_height:
                    end_slice_bottom = int(num_output_pixels - (temp_shift_row_bottom - output_height))

                if temp_shift_col_right > output_width:
                    end_slice_right = int(num_output_pixels - (temp_shift_col_right - output_width))

                output_slice = np.zeros((num_output_pixels, num_output_pixels, 3))
                output_slice[start_slice_top: end_slice_bottom, start_slice_left: end_slice_right] = \
                slide[start_image_top: end_image_bottom, start_image_left: end_image_right]

                if i in all_paths_tumor_str_input:
                    save_path = TUMOR_DIR_OUTPUT
                else:
                    save_path = NO_TUMOR_DIR_OUTPUT

                output_file_name = save_path + 'img_' + str(img_index) + '.jpg'

                try:
                    cv2.imwrite(output_file_name, output_slice)
                except Exception as oerr:
                    print('Error with saving:', oerr)

            except Exception as oerr:
                print('Error with slice:', oerr)


        data_root_tumor_output = pathlib.Path(TUMOR_DIR_OUTPUT)
        all_image_paths_tumor_output = list(data_root_tumor_output.glob('*'))
        all_paths_tumor_str_output = [str(path) for path in all_image_paths_tumor_output]
        num_tumor_images_output = len(all_paths_tumor_str_output)

        data_root_notumor_output = pathlib.Path(NO_TUMOR_DIR_OUTPUT)
        all_image_paths_notumor_output = list(data_root_notumor_output.glob('*'))
        all_paths_notumor_str_output = [str(path) for path in all_image_paths_notumor_output]
        num_notumor_images_output = len(all_paths_notumor_str_output)

        if (num_tumor_images_output != num_tumor_images_input) or (num_notumor_images_output != num_notumor_images_input):
            print('ERROR: Number of output images not the same as number of input images')


# This function is called during testing the slides
def test_part_1(testing_image_path, num_pixels=64, num_level=3):

    slide_path_test = testing_image_path
    tumor_mask_path_test = slide_path_test.split('.')[0]+'_mask.tif'
    print(slide_path_test, tumor_mask_path_test)

    # Retrieve slide parameters before overwriting
    slide, tumor_mask = load_image(slide_path_test, tumor_mask_path_test)
    width, height = slide.level_dimensions[num_level][0], slide.level_dimensions[num_level][1]

    # Read training image at slide level 3
    slide = read_slide(slide,
                       x=0,
                       y=0,
                       level=num_level,
                       width=width,
                       height=height)

    tumor_mask = read_slide(tumor_mask,
                            x=0,
                            y=0,
                            level=num_level,
                            width=width,
                            height=height)

    # Retrieve new array dimensions
    image_depth, image_width = int(np.ceil(slide.shape[0] / num_pixels)), int(np.ceil(slide.shape[1] / num_pixels))

    # Convert the mask from RGB to a black/white binary
    tumor_mask = tumor_mask[:, :, 0]

    # Determine the portions of the image that are tissue
    tissue_pixels = list(find_tissue_pixels(slide))

    # Turn the tissue pixels into a mask
    tissue_regions = apply_mask(slide, tissue_pixels)

    split_image_test(slide, tissue_regions, num_pixels, num_level, slide_path_test)
    return image_depth, image_width, tumor_mask, tissue_regions, slide


# This function is invoked during training phase
def train_part_1(training_image_path, num_pixels, num_level):

    slide_path = training_image_path
    tumor_mask_path = training_image_path.split('.')[0]+'_mask.tif'
    print(slide_path, tumor_mask_path)
    slide, tumor_mask = load_image(slide_path, tumor_mask_path)
    width, height = slide.level_dimensions[num_level][0], slide.level_dimensions[num_level][1]

    slide = read_slide(slide,
                       x=0,
                       y=0,
                       level=num_level,
                       width=width,
                       height=height)

    tumor_mask = read_slide(tumor_mask,
                            x=0,
                            y=0,
                            level=num_level,
                            width=width,
                            height=height)

    # Convert the mask from RGB to a black/white binary
    tumor_mask = tumor_mask[:, :, 0]

    # Determine the portions of the image that are tissue
    tissue_pixels = list(find_tissue_pixels(slide))

    # Turn the tissue pixels into a mask
    tissue_regions = apply_mask(slide, tissue_pixels)

    # Call the split function on the training data
    split_image_and_mask(slide, tumor_mask, tissue_regions, num_pixels, num_level, slide_path)
