"""Processing

Processing utilities for ultrasound data.

Author(s): Tristan Stevens
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import correlate2d

from joint_diffusion.utils.utils import translate


def companding_tf(
    array,
    expand: bool = False,
    comp_type: str = None,
    mu: float = 255,
    A: float = 87.6,
):
    """Companding according to the A- or μ-law algorithm.
    Tensorflow versions of companding.

    Invertible compressing operation. Used to compress
    dynamic range of input data (and subsequently expand).

    μ-law companding:
    https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
    A-law companding:
    https://en.wikipedia.org/wiki/A-law_algorithm

    The μ-law algorithm provides a slightly larger dynamic range
    than the A-law at the cost of worse proportional distortion
    for small signals.

    Args:
    array (ndarray): input array. expected to be in range [-1, 1].
        expand (bool, optional): If set to False (default),
            data is compressed, else expanded.
        comp_type (str): either `a` or `mu`.
        mu (float, optional): compression parameter. Defaults to 255.
        A (float, optional): compression parameter. Defaults to 255.

    Returns:
        ndarray: companded array. has values in range [-1, 1].
    """
    array = tf.clip_by_value(array, -1, 1)
    array = tf.cast(array, tf.float32)
    A = tf.cast(A, tf.float32)
    mu = tf.cast(mu, tf.float32)

    if comp_type is None:
        comp_type = "mu"
    assert comp_type.lower() in ["a", "mu"]

    def mu_law_compress(x):
        y = tf.sign(x) * tf.math.log(1.0 + mu * tf.abs(x)) / tf.math.log(1.0 + mu)
        return y

    def mu_law_expand(y):
        x = tf.sign(y) * ((1 + mu) ** (tf.abs(y)) - 1.0) / mu
        return x

    def a_law_compress(x):
        x_sign = tf.sign(x)
        x_abs = tf.abs(x)
        A_log = tf.math.log(A)

        val1 = x_sign * A * x_abs / (1.0 + A_log)
        val2 = x_sign * (1.0 + tf.math.log(A * x_abs)) / (1.0 + A_log)

        y = tf.where((x_abs >= 0) & (x_abs < (1 / A)), val1, val2)
        return y

    def a_law_expand(y):
        y_sign = tf.sign(y)
        y_abs = tf.abs(y)
        A_log = tf.math.log(A)

        val1 = y_sign * y_abs * (1.0 + A_log) / A
        val2 = y_sign * tf.exp(y_abs * (1.0 + A_log) - 1.0) / A

        x = tf.where((y_abs >= 0) & (y_abs < (1 / (1 + A_log))), val1, val2)
        return x

    if comp_type.lower() == "mu":
        if expand:
            array_out = mu_law_expand(array)
        else:
            array_out = mu_law_compress(array)
    elif comp_type.lower() == "a":
        if expand:
            array_out = a_law_expand(array)
        else:
            array_out = a_law_compress(array)

    return array_out


def companding(image, image_range=(0, 1), expand=False, mu=255):
    """Companding wrapper function to set data in companding range."""
    image = translate(image, image_range, (-1, 1))
    if expand:
        image = companding_tf(image, expand=True, mu=mu)
    else:
        image = companding_tf(image, mu=mu)
    image = translate(image, (-1, 1), image_range)
    return image


def cumulative_distribution(x, n_bins=None):
    """
    Computes the cumulative distribution function (CDF) of a given array of values.

    Args:
        x (array-like): Input array of values.
        n_bins (int): Number of bins to use for the histogram.

    Returns:
        A function that computes the CDF of a given input value.
    """

    def _to_numpy(x):
        if isinstance(x, (float, int)):
            x = [x]
        return np.array(x, dtype=np.float32).flatten()

    x = _to_numpy(x)
    if n_bins is not None:
        hist, bins = np.histogram(x, bins=n_bins)
    else:
        hist, bins = np.histogram(x, bins="auto", density=True)

    cdf = np.cumsum(hist) / np.sum(hist)
    cdf_func = interp1d(
        bins[1:], cdf, kind="linear", bounds_error=False, fill_value=(0, 1)
    )

    def _cdf_func(x):
        x = _to_numpy(x)
        y = cdf_func(x)
        return y

    return _cdf_func


def ks_test(
    sample: np.ndarray, data: np.ndarray, plot: bool = True, n_bins: int = 100
) -> Tuple[float, float]:
    """
    Computes the Kolmogorov-Smirnov test statistic and p-value for the given sample and data.

    Args:
        sample (np.ndarray): The sample data.
        data (np.ndarray): The data to compare against.
        plot (bool, optional): Whether to plot the CDFs. Defaults to True.
        n_bins (int, optional): The number of bins to use for the histogram. Defaults to 100.

    Returns:
        Tuple[float, float]: The KS test statistic and p-value.
    """
    data = data.flatten()
    sample = sample.flatten()

    statistic, pvalue = stats.ks_2samp(sample, data)

    if plot:
        cdf = cumulative_distribution(data)
        hist, bins = np.histogram(sample, bins=n_bins)
        e_cdf = np.cumsum(hist) / np.sum(hist)

        est_cdf = cdf(bins[1:])
        # take argmax ignoring inf and nans
        idx = np.nanargmax(np.abs(est_cdf - e_cdf))
        distance = np.abs(est_cdf - e_cdf)[idx]
        plt.figure()
        plt.plot(bins[1:], est_cdf, label="CDF")
        plt.plot(bins[1:], e_cdf, label="Emperical CDF")
        plt.legend()
        plt.xlabel("Value")
        plt.ylabel("Cumulative Probability")
        plt.title("Kolmogorov-Smirnov Test")
        plt.grid()
        plt.text(bins[1:][idx], est_cdf[idx], f"Distance: {distance:.3f}")
        plt.plot([bins[1:][idx], bins[1:][idx]], [est_cdf[idx], e_cdf[idx]], "r--")

    return statistic, pvalue


def fit_and_get_fwhm(data):
    """Fits a curve to the input data and calculates the
    full width at half maximum (FWHM) of the curve.

    Args:
        data (numpy.ndarray): The input data to fit the curve to.

    Returns:
        float: The FWHM of the curve.
    """
    x = np.arange(len(data))

    # interpolate to find the exact half-maximum value
    interp_factor = 10
    x_interp = np.linspace(0, len(data) - 1, len(data) * interp_factor)
    data_interp = interp1d(x, data, kind="cubic")(x_interp)
    # Find half-maximum value
    max_idx = np.argmax(data_interp)
    max_value = data_interp[max_idx]

    hm_value = max_value / 2
    hm_left_idx = np.argmin(np.abs(data_interp[:max_idx] - hm_value))
    hm_right_idx = np.argmin(np.abs(data_interp[max_idx:] - hm_value)) + max_idx

    # find the original fmhw taken into account the interpolation factor
    hm_left = x_interp[hm_left_idx]
    hm_right = x_interp[hm_right_idx]
    fwhm = hm_right - hm_left

    return fwhm


def get_fhwm_from_autocorrelation(data: np.ndarray) -> Tuple[float, float]:
    """Plots the autocorrelation and full width at half maximum (FWHM)
    of given data.

    Args:
        data (np.ndarray): The input data to plot.
    Returns:
        Tuple[float, float]: The FWHM of the x and y autocorrelation.

    """
    autocorrelation = correlate2d(data, data, mode="same", boundary="wrap")
    autocorrelation = (autocorrelation - np.min(autocorrelation)) / (
        np.max(autocorrelation) - np.min(autocorrelation)
    )
    # Find the maximum value and its position
    y_max, x_max = np.unravel_index(np.argmax(autocorrelation), autocorrelation.shape)

    # Create 1D profiles along x and y directions
    x_profile = autocorrelation[y_max, :]
    y_profile = autocorrelation[:, x_max]

    # Fit the profiles to determine FWHM
    fwhm_x = fit_and_get_fwhm(x_profile)
    fwhm_y = fit_and_get_fwhm(y_profile)

    return fwhm_x, fwhm_y


def histogram_match(image, template=None, histogram=None):
    """Histogram matching

    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Args:
        image (ndarray): Image to transform; the histogram is
            computed over the flattened array.
        template (ndarray): Template image; can have different
            dimensions to source image.
        histogram (tuple): tuple with `values` and `counts`.

    Returns:
        ndarray: The transformed output image
    """
    if not bool(template) ^ bool(histogram):
        raise ValueError(
            "Please provide either a template image or histogram, but not both."
        )

    original_shape = image.shape
    image = image.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    _, bin_idx, s_counts = np.unique(
        image,
        return_inverse=True,
        return_counts=True,
    )

    if histogram is None:
        template = template.ravel()
        t_values, t_counts = np.unique(template, return_counts=True)
    else:
        t_values, t_counts = histogram

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(original_shape)


def equalize_histogram(image, n_bins=256, input_range=None, bi_hist=False):
    """Equalize the histogram of an image (with a uniform target distribution).

    Args:
        image (ndarray): input image / array of any size.
        n_bins (int, optional): number of bins of histogram. Defaults to 256.
        bi_hist (bool, optional): whether to do bi-histogram equalization.
            Defaults to False. In that case vanilla histogram equalization is done.

    Returns:
        ndarray: equalized image / array.
    """

    original_shape = image.shape
    image = image.ravel()

    if input_range is None:
        min_val, max_val = np.min(image), np.max(image)
    else:
        min_val, max_val = input_range

    bins = np.linspace(min_val, max_val, n_bins)

    # Apply either vanilla histogram equalization or bi histogram method
    if bi_hist:
        # Split the histogram of the image into two parts centered around the mean intensity value
        mean = np.mean(image)

        # Divide the bins of the histogram into two parts based on the mean intensity value
        bins_left = bins[: int(mean)]
        bins_right = bins[int(mean) :]

        # Compute the histogram of the image for each histogram part
        image_left = image[image < mean]
        image_right = image[image >= mean]
        hist_left, _ = np.histogram(image_left, bins=bins_left)
        hist_right, _ = np.histogram(image_right, bins=bins_right)

        # Compute the cumulative distribution functions (CDFs) of each histogram part
        cdf_left = np.cumsum(hist_left) / np.sum(hist_left)
        cdf_right = np.cumsum(hist_right) / np.sum(hist_right)

        # scale CDFs
        cdf_left = bins_left[0] + (bins_left[-1] - bins_left[0]) * cdf_left
        cdf_right = bins_right[0] + (bins_right[-1] - bins_right[0]) * cdf_right

        # Compute the lookup table (LUT) for histogram equalization
        lut = np.concatenate((cdf_left, cdf_right))

        # Apply the histogram equalization using the computed LUT
        result = lut[image]

    else:  # vanilla histogram equalization
        # Compute the histogram and cumulative distribution function (CDF) of the image
        hist, bins = np.histogram(image, bins=bins)
        cdf = np.cumsum(hist) / np.sum(hist)

        # scale CDFs
        cdf = bins[0] + (bins[-1] - bins[0]) * cdf

        # Apply the histogram equalization using the computed LUT
        result = cdf[image]

    return result.reshape(original_shape)


def adaptive_equalize_histogram(image, window_size=64, n_bins=256, bi_hist=False):
    """Adaptive (window-based) histogram equalization.

    Args:
        image (ndarray): input image / array of any size.
        window_size (int, optional): size of window. Defaults to 64.
        n_bins (int, optional): number of bins of histogram. Defaults to 256.
        bi_hist (bool, optional): whether to do bi-histogram equalization.
            Defaults to False. In that case vanilla histogram equalization is done.

    Returns:
        ndarray: equalized image / array.
    """
    if isinstance(window_size, int):
        window_size = (window_size, window_size)

    image_size = image.shape
    image_size_y, image_size_x = image_size
    window_size_y, window_size_x = window_size

    # Compute the number of tiles in each dimension
    num_tiles_x = int(np.ceil(image_size_x / window_size_x))
    num_tiles_y = int(np.ceil(image_size_y / window_size_y))

    # Create an empty result image
    result = np.zeros_like(image)

    # Loop over each tile in the image
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            # Compute the coordinates of the current tile
            x1 = j * window_size_x
            x2 = x1 + window_size_x
            y1 = i * window_size_y
            y2 = y1 + window_size_y

            # Check if the current tile extends beyond the image boundary
            x2 = max(x2, image_size_x)
            y2 = max(y2, image_size_y)

            # Extract the current tile
            tile = image[y1:y2, x1:x2]

            # Apply histogram equalization to the current tile using the equalize_histogram function
            equalized_tile = equalize_histogram(tile, bi_hist=bi_hist)

            # Copy the equalized tile to the result image
            result[y1:y2, x1:x2] = equalized_tile

    return result


def calculate_intensity_threshold(
    image: np.ndarray, percentile: int
) -> Tuple[int, float]:
    """
    Calculate intensity threshold based on top percentile values in the image and
    return the mean value of the pixels that exceed the threshold.

    Args:
        image: Grayscale image as a numpy array.
        percentile: The top percentile values to consider.

    Returns:
        A tuple containing the intensity threshold value and the mean value of
        the pixels that exceed the threshold.
    """
    threshold = np.percentile(image, 100 - percentile)
    above_threshold = image[image >= threshold]
    mean_above_threshold = np.mean(above_threshold)
    return threshold, mean_above_threshold


def gcnr(x, y, bins=256):
    """Generalized contrast-to-noise-ratio"""
    x = x.flatten()
    y = y.flatten()
    _, bins = np.histogram(np.concatenate((x, y)), bins=bins)
    f, _ = np.histogram(x, bins=bins, density=True)
    g, _ = np.histogram(y, bins=bins, density=True)
    f /= f.sum()
    g /= g.sum()
    return 1 - np.sum(np.minimum(f, g))
