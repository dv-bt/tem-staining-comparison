"""
Functions for calculating the Fourier Transform of images and their power spectrum.
"""

import numpy as np
import scipy.fft as fft
import scipy.ndimage as ndimage


def power_spectrum(image: np.ndarray, log: bool = True) -> np.ndarray:
    """Calculate the power spectrum of an image.

    Parameters
    ----------
    image : np.ndarray
        The image to calculate the power spectrum of.
    log : bool, optional
        Whether to take the base 10 log of the power spectrum, by default True.

    Returns
    -------
    f_magnitude : np.ndarray
        The power spectrum of the image.
    """
    f_transform = fft.fft2(image)
    f_magnitude = np.abs(fft.fftshift(f_transform)) ** 2

    if log:
        f_magnitude = np.log10(f_magnitude)

    return f_magnitude


def radially_averaged_spectrum(image: np.ndarray, log: bool = True) -> np.ndarray:
    """Calculate the radial average of the power spectrum of an image.

    Parameters
    ----------
    image : np.ndarray
        The image to calculate the power spectrum of.
    log : bool, optional
        Whether to take the base 10 log of the power spectrum, by default True.

    Returns
    -------
    f_magnitude : np.ndarray
        The power spectrum of the image.
    """

    f_magnitude = power_spectrum(image, log=log)

    # Create a grid of distances from the center
    grid = np.indices((image.shape[0], image.shape[1]))
    center = np.array([image.shape[0] // 2, image.shape[1] // 2])
    distances = np.linalg.norm(grid - center[:, None, None], axis=0)

    # Bin the power spectrum values by their radial distance
    distances = distances.astype(int)
    radial_mean = ndimage.mean(
        f_magnitude, labels=distances, index=np.arange(0, distances.max() + 1)
    )

    return radial_mean
