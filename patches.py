"""Patches

Tensorflow utilities for creating patches from images and stitching patches back together.

Author(s): Tristan Stevens
"""

from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from scipy.signal import hann


def check_patches_fit(
    image_shape: tuple, patch_shape: tuple, overlap: Union[int, Tuple[int, int]]
) -> tuple:
    """Checks if patches with overlap fit an integer amount in the original image.

    Args:
        image_shape: A tuple representing the shape of the original image.
        patch_size: A tuple representing the shape of the patches.
        overlap: A float representing the overlap between patches.

    Returns:
        A tuple containing a boolean indicating if the patches fit an integer amount
        in the original image and the new image shape if the patches do not fit.
    """
    if overlap:
        stride = (np.array(patch_shape) - np.array(overlap)).astype(int)
    else:
        stride = (np.array(patch_shape)).astype(int)
        overlap = (0, 0)

    stride_y, stride_x = stride
    patch_y, patch_x = patch_shape
    image_y, image_x = image_shape
    overlap_y, overlap_x = overlap

    if (image_y - patch_y) % stride_y != 0 or (image_x - patch_x) % stride_x != 0:
        new_shape = (
            (image_y - patch_y) // stride_y * stride_y + patch_y,
            (image_x - patch_x) // stride_x * stride_x + patch_x,
        )
        print(
            f"Warning: patches with overlap do not fit an integer amount in the original image. "
            f"Cropping image to closest dimensions that work: {new_shape}."
        )

        n_patch_y = image_y // stride_y
        n_patch_x = image_x // stride_x

        new_patch_shape = (
            (image_y + (n_patch_y - 1) * overlap_y) / n_patch_y,
            (image_x + (n_patch_x - 1) * overlap_x) / n_patch_x,
        )

        print(f"Alternatively, change patch shape to: {new_patch_shape} ")

        return False, new_shape
    return True, image_shape


def generate_window(
    row, column, image_size_y, image_size_x, patch_shape, overlap, window_type
):
    """Generates a window for a patch.

    Args:
        row (int): Row index of the patch.
        column (int): Column index of the patch.
        image_size_y (int): Height of the output image.
        image_size_x (int): Width of the output image.
        patch_shape (Tuple): Shape of the patch. (y, x, n_channels) or (y, x)
        overlap (float): Overlap factor between patches.
        window_type (str): Window type. 'hann' or 'average'.

    Returns:
        window (ndarray): Window for the patch.

    """
    patch_size_y, patch_size_x = patch_shape[:2]
    window = np.ones(patch_shape)

    # Calculate overlap for each edge of the patch
    overlap_y = int(patch_size_y * overlap)
    overlap_x = int(patch_size_x * overlap)

    # Generate the window
    if window_type == "hann":
        window[:overlap_y, :] *= hann(overlap_y * 2)[:overlap_y, np.newaxis, np.newaxis]
        window[-overlap_y:, :] *= hann(overlap_y * 2)[
            -overlap_y:, np.newaxis, np.newaxis
        ]
        window[:, :overlap_x] *= hann(overlap_x * 2)[np.newaxis, :overlap_x, np.newaxis]
        window[:, -overlap_x:] *= hann(overlap_x * 2)[
            np.newaxis, -overlap_x:, np.newaxis
        ]
    elif window_type == "average":
        window[:overlap_y, :] *= 0.5
        window[-overlap_y:, :] *= 0.5
        window[:, :overlap_x] *= 0.5
        window[:, -overlap_x:] *= 0.5
    else:
        raise ValueError(f"Invalid window type: {window_type}")

    # Adjust the window for patches on the edge of the image
    # top left
    if row == 0 and column == 0:
        window[:overlap_y, :] *= 2
        window[:, :overlap_x] *= 2
    # top right
    elif row == 0 and column + patch_size_x == image_size_x:
        window[:overlap_y, :] *= 2
        window[:, -overlap_x:] *= 2
    # bottom left
    elif row + patch_size_y == image_size_y and column == 0:
        window[-overlap_y:, :] *= 2
        window[:, :overlap_x] *= 2
    # bottom right
    elif row + patch_size_y == image_size_y and column + patch_size_x == image_size_x:
        window[-overlap_y:, :] *= 2
        window[:, -overlap_x:] *= 2
    # top edge
    elif row == 0:
        window[:overlap_y, :] *= 2
    # bottom edge
    elif row + patch_size_y == image_size_y:
        window[-overlap_y:, :] *= 2
    # left edge
    elif column == 0:
        window[:, :overlap_x] *= 2
    # right edge
    elif column + patch_size_x == image_size_x:
        window[:, -overlap_x:] *= 2

    return window


def images_to_patches_tf(
    images: tf.Tensor,
    patch_shape: Union[int, Tuple[int, int]],
    overlap: Union[int, Tuple[int, int]] = None,
) -> tf.Tensor:
    """Creates patches from images.

    Args:
        images (Tensor): input images [batch, height, width, channels].
        patch_shape (int or tuple, optional): Height and width of patch. Defaults to 4.
        overlap (int or tuple, optional): Overlap between patches in px. Defaults to None.

    Returns:
        patches (Tensor): batch of patches of size:
            [batch, #patch_y, #patch_x, patch_size_y, patch_size_x, #channels].

    """
    assert (
        len(images.shape) == 4
    ), f"input array should have 4 dimensions, but has {len(images.shape)} dimensions"
    assert (
        isinstance(patch_shape, int) or len(patch_shape) == 2
    ), f"patch_shape should be an integer or a tuple of length 2, but is {patch_shape}"
    assert (
        isinstance(overlap, (int, type(None))) or len(overlap) == 2
    ), f"overlap should be an integer or a tuple of length 2, but is {overlap}"

    batch_size, *image_shape, n_channels = images.shape

    if isinstance(patch_shape, int):
        patch_shape = (patch_shape, patch_shape)
    if isinstance(overlap, int):
        overlap = (overlap, overlap)

    patch_size_y, patch_size_x = patch_shape

    patches_fit, image_shape = check_patches_fit(image_shape, patch_shape, overlap)
    if not patches_fit:
        images = images[:, : image_shape[0], : image_shape[1], :]

    if overlap:
        stride = (np.array(patch_shape) - np.array(overlap)).astype(int)
    else:
        stride = np.array(patch_shape).astype(int)

    # add channel dimension to stride / patch_shape
    stride = [1, *stride, 1]
    patch_shape = [1, *patch_shape, 1]

    # create patches
    patches = tf.image.extract_patches(
        images, sizes=patch_shape, strides=stride, rates=[1, 1, 1, 1], padding="VALID"
    )

    _, n_patch_y, n_patch_x, *_ = patches.shape

    shape = [batch_size, n_patch_y, n_patch_x, patch_size_y, patch_size_x, n_channels]
    patches = tf.reshape(patches, shape)
    return patches


def patches_to_images_tf(
    patches: np.ndarray,
    image_shape: tuple,
    overlap: Union[int, Tuple[int, int]] = None,
    window_type="replace",
    indices=None,
) -> np.ndarray:
    """Reconstructs images from patches.

    Args:
        patches (ndarray): Array with batch of patches to convert to batch of images.
            [batch_size, #patch_y, #patch_x, patch_size_y, patch_size_x, n_channels]
        image_shape (Tuple): Shape of output image. (y, x, n_channels) or (y, x)
        overlap (int or tuple, optional): Overlap between patches in px. Defaults to None.
        window_type (str, optional): Type of stitching to use. Defaults to 'hann'.
            Options: 'hann', 'average', 'replace'.
        indices (ndarray, optional): Indices of patches in image. Defaults to None.
            If provided, indices are used to stitch patches together and not recomputed
            to save time. Has same shape as patches shape but with added index axis (last).
    Returns:
        images (ndarray): Reconstructed batch of images from batch of patches.

    """
    assert len(image_shape) == 3, (
        "image_shape should have 3 dimensions, namely: "
        "(#image_y, #image_x, (n_channels))"
    )
    assert len(patches.shape) == 6, (
        "patches should have 6 dimensions, namely: "
        "[batch, #patch_y, #patch_x, patch_size_y, patch_size_x, n_channels]"
    )
    assert (
        isinstance(overlap, (int, type(None))) or len(overlap) == 2
    ), f"overlap should be an integer or a tuple of length 2, but is {overlap}"
    assert window_type in ["hann", "average", "replace"], (
        "window_type should be either " '"hann", "average" or "replace"'
    )

    batch_size, n_patches_y, n_patches_x, *patch_shape = patches.shape
    n_channels = image_shape[-1]
    dtype = patches.dtype

    patches_fit, new_image_shape = check_patches_fit(
        image_shape[:2], patch_shape[:2], overlap
    )
    if not patches_fit:
        image_shape = (*new_image_shape, image_shape[-1])

    assert len(patch_shape) == len(image_shape)

    # Kernel for counting overlaps
    if window_type == "average":
        kernel_ones = tf.ones(
            (batch_size, n_patches_y, n_patches_x, *patch_shape), dtype=tf.int32
        )
        mask = tf.zeros((batch_size, *image_shape), dtype=tf.int32)
    elif window_type == "hann":
        hann_window = hann(patch_shape[0])[:, np.newaxis] * hann(patch_shape[1])
        # slightly change hann such that there are no zeros
        hann_window[hann_window == 0] = 1e-8
        hann_window = hann_window[np.newaxis, ..., np.newaxis]
        hann_window = tf.cast(hann_window, dtype)
        patches = patches * hann_window
        mask = tf.zeros((batch_size, *image_shape), dtype=dtype)

    if indices is None:
        if overlap:
            if isinstance(overlap, int):
                overlap = (overlap, overlap)
            overlap = [*overlap, 1]
            stride = (np.array(patch_shape) - np.array(overlap)).astype(int)
        else:
            stride = (np.array(patch_shape)).astype(int)

        channel_idx = tf.reshape(tf.range(n_channels), (1, 1, 1, 1, 1, n_channels, 1))
        channel_idx = (
            tf.ones(
                (batch_size, n_patches_y, n_patches_x, *patch_shape, 1), dtype=tf.int32
            )
            * channel_idx
        )

        batch_idx = tf.reshape(tf.range(batch_size), (batch_size, 1, 1, 1, 1, 1, 1))
        batch_idx = (
            tf.ones(
                (batch_size, n_patches_y, n_patches_x, *patch_shape, 1), dtype=tf.int32
            )
            * batch_idx
        )

        # TODO: create indices without looping possibly
        indices = []
        for j in range(n_patches_y):
            for i in range(n_patches_x):
                # Make indices from meshgrid
                _indices = tf.meshgrid(
                    tf.range(
                        stride[0] * j, patch_shape[0] + stride[0] * j  # row start
                    ),  # row end
                    tf.range(
                        stride[1] * i, patch_shape[1] + stride[1] * i  # col_start
                    ),
                    indexing="ij",
                )  # col_end

                _indices = tf.stack(_indices, axis=-1)
                indices.append(_indices)

        indices = tf.reshape(
            tf.stack(indices, axis=0), (n_patches_y, n_patches_x, *patch_shape[:2], 2)
        )

        indices = tf.repeat(indices[tf.newaxis, ...], batch_size, axis=0)
        indices = tf.repeat(indices[..., tf.newaxis, :], n_channels, axis=-2)

        indices = tf.concat([batch_idx, indices, channel_idx], axis=-1)

    # create output image tensor
    images = tf.zeros([batch_size, *image_shape], dtype=dtype)

    # Add sliced image to recovered image indices
    if window_type == "replace":
        images = tf.tensor_scatter_nd_update(images, indices, patches)
    elif window_type == "average":
        images = tf.tensor_scatter_nd_add(images, indices, patches)
        mask = tf.tensor_scatter_nd_add(mask, indices, kernel_ones)
        images = tf.cast(images, tf.float32) / tf.cast(mask, tf.float32)
    if window_type == "hann":
        images = tf.tensor_scatter_nd_add(images, indices, patches)
        hann_window = tf.tile(
            tf.reshape(hann_window, (1, 1, 1, *patch_shape[:2], 1)),
            (batch_size, n_patches_y, n_patches_x, 1, 1, n_channels),
        )
        mask = tf.tensor_scatter_nd_add(mask, indices, hann_window)
        images = tf.cast(images, tf.float32) / tf.cast(mask, tf.float32)

    return images, indices
