"""
Created on Fri Oct  4 00:44:31 2024

@author: Hashir Irfan
"""

import itertools

import cv2 as cv
import imageio as IO
import matplotlib.pyplot as plt
import numpy as np

cv_thresholds = {
    "BINARY": cv.THRESH_BINARY,
    "BINARY_INV": cv.THRESH_BINARY_INV,
    "TRUC": cv.THRESH_TRUNC,
    "TO_ZERO": cv.THRESH_TOZERO,
    "TO_ZERO_INV": cv.THRESH_TOZERO_INV,
}


def read_image(filename: str):
    """Read an image from a file.

    This function uses the `IO.imread` method to load an image from the specified filename. It returns the image data for further processing or analysis.

    Args:
        filename (str): The path to the image file to be read.

    Returns:
        ndarray: The image data as an array.
    """
    return IO.imread(filename)


def read_image_by_cv2(filename: str, to_gray_scale: int = 0):
    """
    Reads an image from a specified file using OpenCV. This function allows the option to convert the image to grayscale.

    It utilizes OpenCV's `imread` function to load the image from the given filename. The `to_gray_scale` parameter determines whether the image should be read in color or converted to grayscale.

    Args:
        filename (str): The path to the image file to be read.
        to_gray_scale (int, optional): A flag indicating whether to convert the image to grayscale (1) or read it in color (0). Defaults to 0.

    Returns:
        numpy.ndarray: The image read from the file, represented as a NumPy array.
    """
#    return cv.imread(filename, cv.IMREAD_GRAYSCALE)
    return cv.imread(filename)


def resize_image(image, rows: int, columns: int):
    """
    Resizes an image to the specified dimensions. This function allows for changing the size of an image by providing the desired number of rows and columns.

    It utilizes OpenCV's `resize` function to adjust the dimensions of the input image. The resized image will have the specified number of rows and columns.

    Args:
        image (numpy.ndarray): The image to be resized, represented as a NumPy array.
        rows (int): The desired number of rows for the resized image.
        columns (int): The desired number of columns for the resized image.

    Returns:
        numpy.ndarray: The resized image represented as a NumPy array.
    """
    return cv.resize(image, (rows, columns))


def display_image_by_cv2(label: str, image):
    """
    Displays an image in a window using OpenCV. This function shows the provided image with an associated label and waits for a key press before closing the window.

    It utilizes OpenCV's `imshow` function to create a window displaying the image. The window remains open until a key is pressed, after which all OpenCV windows are destroyed.

    Args:
        image (numpy.ndarray): The image to be displayed, represented as a NumPy array.
        label (str): The title of the window in which the image will be displayed.

    Returns:
        None
    """
    cv.imshow(label, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def segmentation_by_cv2(
    image, threshold: float, max_value: int, segmentation_type: str
):
    """
    Applies image segmentation using OpenCV's thresholding methods. This function segments the input image based on a specified threshold and segmentation type.

    It utilizes OpenCV's `threshold` function to classify pixel values in the image. The result is a binary or segmented image based on the provided threshold, maximum value, and segmentation type.

    Args:
        image (numpy.ndarray): The input image to be segmented, represented as a NumPy array.
        threshold (float): The threshold value used for segmentation.
        max_value (int): The maximum value to use with the thresholding operation.
        segmentation_type (str): The type of thresholding to apply (e.g., 'BINARY').

    Returns:
        numpy.ndarray: The segmented image resulting from the thresholding operation.
    """
    return cv.threshold(image, threshold, max_value, cv_thresholds[segmentation_type])


def otsu_segmentation_by_cv2(image, max_value: int, segmentation_type: str):
    """
    Applies image segmentation using OpenCV's thresholding methods. This function segments the input image based on a specified threshold and segmentation type.

    It utilizes OpenCV's `threshold` function to classify pixel values in the image. The result is a binary or segmented image based on the provided threshold, maximum value, and segmentation type.

    Args:
        image (numpy.ndarray): The input image to be segmented, represented as a NumPy array.
        max_value (int): The maximum value to use with the thresholding operation.
        segmentation_type (str): The type of thresholding to apply (e.g., 'BINARY').

    Returns:
        numpy.ndarray: The segmented image resulting from the thresholding operation.
    """
    return cv.threshold(
        image, 0, max_value, cv_thresholds[segmentation_type] + cv.THRESH_OTSU
    )


def adaptive_threshold(
    image,
    max_value: int,
    threshold_type: str,
    block_size: int,
    constant: int,
    adaptive_type: str,
):
    """
    Applies adaptive thresholding to the given image to convert it to a binary image.
    This technique adjusts the threshold value on a per-pixel basis based on the local neighborhood of each pixel.

    Args:
        image: The input image to which adaptive thresholding will be applied.
        max_value (int): The maximum value to use with the binary thresholding.
        threshold_type (str): The type of thresholding to apply (e.g., cv.THRESH_BINARY).
        block_size (int): The size of the neighborhood area used to calculate the threshold.
        constant (int): A constant subtracted from the mean or weighted mean.

    Returns:
        The binary image resulting from the adaptive thresholding operation.
    """
    return cv.adaptiveThreshold(
        image,
        max_value,
        adaptive_type,
        cv_thresholds[threshold_type],
        block_size,
        constant,
    )


def display_image(im, is_gray: bool = True, show_axis: bool = False) -> None:
    """Display an image using Matplotlib.

    This function visualizes an image with optional grayscale and axis display settings. It uses Matplotlib to render the image and can toggle the visibility of the axes based on the provided parameters.

    Args:
        im (array-like): The image data to be displayed.
        is_gray (bool, optional): If True, the image will be displayed in grayscale. Defaults to True.
        show_axis (bool, optional): If True, the axes will be shown. Defaults to False.

    Returns:
        None
    """
    plt.imshow(im, cmap="gray" if is_gray else None)
    plt.axis("on" if show_axis else "off")
    plt.show()
    plt.close()


def get_shape(im):
    """Retrieve the shape of an image.

    This function returns the dimensions of the provided image as a tuple. It is useful for understanding the structure of the image data for further processing.

    Args:
        im (array-like): The image whose shape is to be retrieved.

    Returns:
        tuple: The shape of the image as a tuple representing its dimensions.
    """
    return im.shape


def get_meta_data(im) -> dict:
    """Retrieve the metadata of an image.

    This function returns the metadata associated with the provided image. It is useful for accessing additional information about the image, such as its format, size, and other properties.

    Args:
        im (object): The image from which to retrieve metadata.

    Returns:
        object: The metadata of the image.
    """
    return im.meta


def print_image(image, rows: int, cols: int) -> None:
    """Print the pixel values of an image in a formatted grid.

    This function displays the pixel values of the provided image in a specified number of rows and columns. It iterates through the image data and prints each pixel value, ensuring a clear visual representation of the image structure.

    Args:
        image (array-like): The image data to be printed.
        rows (int): The number of rows to display.
        cols (int): The number of columns to display.

    Returns:
        None
    """
    for r in range(rows):
        print(" ".join(str(image[r, c]) for c in range(cols)))


def get_meta_data_keys_value(image, keys: list) -> dict:
    """Retrieve specified metadata keys and their values from an image.

    This function extracts the metadata from the given image and returns a dictionary containing only the specified keys and their corresponding values. If a key is not present in the metadata, it will be omitted from the returned dictionary.

    Args:
        image (object): The image from which to retrieve metadata.
        keys (list): A list of keys for which to retrieve the corresponding metadata values.

    Returns:
        dict: A dictionary containing the specified metadata keys and their values.
    """
    meta_data = get_meta_data(image)
    return {key: meta_data[key] for key in keys if key in meta_data}


def convert_color_image_to_grey_scale(image, rows: int, columns: int) -> list:
    """Convert a color image to grayscale.

    This function takes a color image and transforms it into a grayscale image by applying a weighted sum to the RGB values of each pixel. The resulting grayscale values replace the original color values in the image.

    Args:
        image (array-like): The input color image to be converted.
        rows (int): The number of rows in the image.
        columns (int): The number of columns in the image.

    Returns:
        None: The function modifies the input image in place.
    """
    new_image = np.zeros((rows, columns))
    for r, c in itertools.product(range(rows), range(columns)):
        new_image[r][c] = (
            (image[r, c, 0] * 0.3) + (image[r, c, 1] * 0.59) + (image[r, c, 2] * 0.11)
        )
    return new_image


def construct_histogram(image, bins: int = 100, is_ravel: bool = True) -> None:
    """
    Generate and display a histogram for the provided image data. This function visualizes the distribution of pixel intensities.

    The histogram can be customized by specifying the number of bins and whether to flatten the image array before plotting.
    It is useful for analyzing the tonal range and contrast of the image.

    Args:
        image (array-like): The input image data to be visualized.
        bins (int, optional): The number of bins for the histogram. Defaults to 100.
        is_ravel (bool, optional): If True, the image will be flattened before plotting. Defaults to True.

    Returns:
        None: This function does not return a value; it displays the histogram plot.
    """
    plt.hist(image.ravel() if is_ravel else image, bins=bins)
    plt.show()
    plt.close()


def equalize_histogram(image: np.array) -> np.array:
    """
    Enhances the contrast of the given image by applying histogram equalization.
    This technique redistributes the intensity values of the image to improve its overall contrast.

    Args:
        image: The input image for which histogram equalization will be applied.

    Returns:
        A new image with enhanced contrast resulting from the histogram equalization process.
    """
    return cv.equalizeHist(image)


def segmentation(image, threshold: float, reverse: bool = False):
    """
    Perform binary segmentation on the input image based on a specified threshold.
    This function creates a new image where pixels above the threshold are set to white, and those below are set to black.

    The segmentation process helps in isolating objects or regions of interest in the image by converting it into a binary format.
    The resulting image can be used for further analysis or processing.

    Args:
        image (array-like): The input image data to be segmented.
        threshold (float): The threshold value used for segmentation. Pixels with values greater than or equal to this will be set to white.

    Returns:
        new_image (array-like): A binary image where pixels are set to 255 for values above the threshold and 0 otherwise.
    """
    size = get_shape(image)
    rows, columns = size[0], size[1]

    new_image = np.zeros((rows, columns), np.uint8)

    for i in range(rows):
        for j in range(columns):
            if reverse:
                new_image[i, j] = 255 if image[i, j] <= threshold else 0
            else:
                new_image[i, j] = 255 if image[i, j] >= threshold else 0
    return new_image


def crop_image(
    image,
    start_row: float,
    row_crop_width: float,
    start_column: float,
    column_crop_height: float,
    rows: int,
    columns: int,
):
    """
   Crops a specified region from the input image based on the provided fractional parameters.
   The function calculates the starting position and dimensions of the crop relative to the image size.

   Args:
       image: The input image to be cropped.
       start_row (float): The starting row position as a fraction of the total rows.
       row_crop_width (float): The width of the crop as a fraction of the total rows.
       start_column (float): The starting column position as a fraction of the total columns.
       column_crop_height (float): The height of the crop as a fraction of the total columns.
       rows (int): The total number of rows in the image.
       columns (int): The total number of columns in the image.

   Returns:
       The cropped portion of the image.

   Raises:
       IndexError: If the calculated crop dimensions exceed the image boundaries.
   """
    crop_start_x = int(start_row * rows)  # Start x at 20% of the width
    crop_start_y = int(start_column * columns)  # Start y at 20% of the height
    crop_width = int(row_crop_width * rows)  # Crop width to 60% of the original width
    crop_height = int(
        column_crop_height * columns
    )  # Crop height to 70% of the original height

    return image[
        crop_start_y : crop_start_y + crop_height,
        crop_start_x : crop_start_x + crop_width,
    ]


import numpy as np


def place_mask_in_image(
    image,
    mask,
    start_row: float,
    row_crop_width: float,
    start_column: float,
    column_crop_height: float,
    rows: int,
    columns: int,
):
    """
    Places a mask back into the original RGB image at the position where the image was cropped.

    Args:
        image (numpy.ndarray): The original RGB image in which the mask will be placed.
        mask (numpy.ndarray): The binary mask (0's and 1's) to be placed into the image.
                              The mask should be 2D, but will be applied across all channels.
        start_row (float): The starting row position as a fraction of the total rows.
        row_crop_width (float): The width of the crop as a fraction of the total rows.
        start_column (float): The starting column position as a fraction of the total columns.
        column_crop_height (float): The height of the crop as a fraction of the total columns.
        rows (int): The total number of rows in the original image.
        columns (int): The total number of columns in the original image.

    Returns:
        numpy.ndarray: The original image with the mask placed back in the corresponding cropped area.

    Raises:
        IndexError: If the mask dimensions do not match the crop area or exceed the image boundaries.
    """
    # Calculate the start position and dimensions of the crop area
    crop_start_x = int(start_row * rows)
    crop_start_y = int(start_column * columns)
    crop_height, crop_width = mask.shape

    # Create a black image of the same size as the original image
    black_image = np.zeros_like(image)

    # Broadcast the mask to 3 channels (R, G, B)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    print(black_image.shape, image.shape, crop_height, crop_width)
    # Apply the mask to the corresponding region of the black image
    black_image[
        crop_start_y : crop_start_y + crop_height,
        crop_start_x : crop_start_x + crop_width,
    ] = np.where(
        mask_3d == 255,
        image[
            crop_start_y : crop_start_y + crop_height,
            crop_start_x : crop_start_x + crop_width,
        ],
        0,
    )

    return black_image


def convolution(image, window: list, _type: str = None):
    """Applies a convolution operation to an image using a specified window (kernel).

    Supports grayscale and color images. For color images, applies the convolution
    separately to each channel.

    Args:
        image: The input image (grayscale or color) to which the convolution will be applied.
        window (list): A 2D list representing the convolution kernel, which must have odd dimensions.
        _type (str, optional): The type of convolution to perform. If set to "PREWIT", applies Prewitt convolution; otherwise, performs standard averaging.

    Returns:
        numpy.ndarray: The resulting image after applying the convolution.
    """
    if not window or len(window) % 2 == 0 or len(window[0]) % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # Convert image to float for calculations
    image = image.astype(np.float32)

    # Handle color or grayscale images
    if len(image.shape) == 3:  # Color image
        channels = image.shape[2]
        result_image = np.zeros_like(image, dtype=np.float32)
        for ch in range(channels):
            result_image[:, :, ch] = convolution(image[:, :, ch], window, _type)
        return result_image

    elif len(image.shape) == 2:  # Grayscale image
        rows, columns = image.shape
        window_size = len(window)
        offset = window_size // 2  # Offset for the window center

        # Pad the image
        padded_image = np.pad(image, offset, mode="edge")

        smoothed_image = np.zeros((rows, columns), dtype=np.float32)
        for r in range(1, rows + 1):
            for c in range(1, columns + 1):
                smoothed_value = 0
                for i in range(window_size):
                    for j in range(window_size):
                        smoothed_value += (
                            padded_image[r - offset + i, c - offset + j] * window[i][j]
                        )

                if _type in ["PREWIT", "LAPLACIAN", "SOBEL"]:
                    smoothed_image[r - 1, c - 1] = smoothed_value
                else:
                    smoothed_image[r - 1, c - 1] = smoothed_value / (window_size ** 2)

        return smoothed_image
    else:
        raise ValueError("Unsupported image format. Must be grayscale or color.")




"""def convolution(image, window: list, _type: str = None):
    Applies a convolution operation to an image using a specified window (kernel).

    This function performs convolution on the input image with the provided window.
    It supports different types of convolution based on the `_type` parameter, allowing for various smoothing effects.

    Args:
        image: The input image to which the convolution will be applied.
        window (list): A 2D list representing the convolution kernel, which must have odd dimensions.
        _type (str, optional): The type of convolution to perform. If set to "PREWIT", it applies Prewitt convolution; otherwise, it performs standard averaging.

    Returns:
        numpy.ndarray: The resulting image after applying the convolution.

    Raises:
        ValueError: If the window is empty or has even dimensions.

    image = image.astype(np.float32)

    if not window or len(window) % 2 == 0 or len(window[0]) % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    size = get_shape(image)
    print(size)
    rows = size[0]
    columns = size[1]

    window_size = len(window)
    offset = window_size // 2  # Offset for the window center

    padded_image = np.pad(image, offset, mode="edge")
    padded_image = padded_image.astype(np.float32)

    #    display_image(padded_image)
    smoothed_image = np.zeros((rows, columns))
    for r in range(1, rows + 1):
        for c in range(1, columns + 1):
            smoothed_value = 0
            for i in range(window_size):
                for j in range(window_size):
                    smoothed_value += padded_image[r - offset + i, c - offset + j] * window[i][j]
#            smoothed_value = sum(
#                (padded_image[r - offset + i, c - offset + j] * window[i][j])
#                for i, j in itertools.product(range(window_size), range(window_size))
#            )
#            print(smoothed_value)
            if _type == "PREWIT":
                smoothed_image[r - 1, c - 1] = smoothed_value
            else:
                smoothed_image[r - 1, c - 1] = smoothed_value // (window_size ** 2)

    return smoothed_image
"""