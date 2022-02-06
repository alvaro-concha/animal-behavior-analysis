"""Watershed segmentation."""
from copy import deepcopy
import numpy as np
from fastkde import fastKDE
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from skimage.filters import gaussian
from skimage.filters.rank import median, gradient
from skimage.morphology import disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, mark_boundaries
from skimage.color import rgb2gray
from scipy.ndimage import label as ndi_label
from utilities import read_pickle, write_pickle
import config


def get_dynamic_range(img):
    """
    Returns image with full dynamic range.

    Parameters
    ----------
    img: array_like
        Image whose intensities are to be rescaled to full dynamic range
    """
    img = (img - img.min()) / (img.max() - img.min())
    img = img_as_ubyte(img)
    return img


def get_wavelets_gaussian_convolution(img):
    """Convolve wavelet outcomes density with gaussian kernel."""
    sigma = (
        config.WAT_EMB_WAV_SIGMA
        * config.WAT_RESOLUTION
        / (config.WAT_EMB_WAV_LIM[1] - config.WAT_EMB_WAV_LIM[0])
    )
    img_smooth = gaussian(img, sigma=sigma)
    img_smooth /= np.sum(img_smooth)
    return img_smooth


def get_umap_wavelets_outcomes_densities(path_out_wavs, path_kde, path_gau_kde):
    """Probability density estimations for UMAP wavelet outcomes."""
    out_wavs = read_pickle(path_out_wavs)
    axes_wavs = [
        np.linspace(*config.WAT_EMB_WAV_LIM, config.WAT_RESOLUTION + 1)
        for _ in range(2)
    ]
    kde_wavs, _ = fastKDE.pdf(
        out_wavs[:, 0],
        out_wavs[:, 1],
        axes=axes_wavs,
    )
    kde_wavs = get_dynamic_range(kde_wavs)
    write_pickle(kde_wavs, path_kde)
    gau_kde_wavs = get_wavelets_gaussian_convolution(kde_wavs)
    write_pickle(gau_kde_wavs, path_gau_kde)


def get_wavelets_watershed_segmentation(out_wavs, img, path_labels, path_segmentation):
    """Performs watershed segmentation over UMAP wavelet outcomes.

    img is a raw KDE, without gaussian smoothing.
    """

    print("1. Obtaining Gaussian KDE of the UMAP out_wavs")
    img = get_wavelets_gaussian_convolution(img)
    img = get_dynamic_range(img)
    plt.imshow(img, origin="lower")
    plt.savefig("gau_kde_wav_all.png", bbox_inches="tight", dpi=400)
    plt.close()

    print("2. Apply denoising median filter, with small window area")
    img = median(img, disk(config.WAT_RADIUS_DENOISE))
    img = get_dynamic_range(img)
    plt.imshow(img, origin="lower")

    print("3. Get image gradient to use as input for watershed segmentation")
    img_gradient = gradient(img, disk(config.WAT_RADIUS_GRADIENT))
    img_gradient = get_dynamic_range(img_gradient)
    plt.imshow(img_gradient, origin="lower")
    plt.savefig("gra_gau_kde_wav_all.png", bbox_inches="tight", dpi=400)
    plt.close()

    print("4. Compute watershed segmentation seeds, local maxima markers")
    img_cut = deepcopy(img)
    img_cut[img < config.WAT_CUT_THRESHOLD] = 0
    local_max_idx = peak_local_max(
        image=img_cut, footprint=disk(config.WAT_RADIUS_FOOTPRINT)
    )
    local_max = np.zeros_like(img, dtype=bool)
    local_max[tuple(local_max_idx.T)] = True
    plt.imshow(img_cut, origin="lower")
    plt.savefig("cut_gau_kde_wav_all.png", bbox_inches="tight", dpi=400)
    plt.close()
    del img_cut

    markers = ndi_label(local_max)[0]
    del local_max

    print("5. Compute compact watershed segmentation")
    predictions = watershed(image=img_gradient, markers=markers, compactness=1)
    segmentation = mark_boundaries(
        image=img,
        label_img=predictions,
        color=(0, 0, 0),
        outline_color=(1, 1, 1),
    )
    segmentation = rgb2gray(segmentation)
    write_pickle(segmentation, path_segmentation)
    out_wavs_trans = deepcopy(out_wavs)

    print("6. Normalize and scale UMAP out_wavs to plot them over the image")
    out_wavs_trans[:, 0] = (
        (out_wavs[:, 0] - config.WAT_EMB_WAV_LIM[0])
        / (config.WAT_EMB_WAV_LIM[1] - config.WAT_EMB_WAV_LIM[0])
        * config.WAT_RESOLUTION
    )
    out_wavs_trans[:, 1] = (
        (out_wavs[:, 1] - config.WAT_EMB_WAV_LIM[0])
        / (config.WAT_EMB_WAV_LIM[1] - config.WAT_EMB_WAV_LIM[0])
        * config.WAT_RESOLUTION
    )

    print("7. Assign watershed segmentation labels to UMAP out_wavs")
    labels = np.array(
        [predictions[int(outcome[1]), int(outcome[0])] for outcome in out_wavs_trans],
        dtype=int,
    )
    del out_wavs_trans
    write_pickle(labels, path_labels)
    print(f"Number of labels: {labels.max()}")


# Here come Steps and Poses:


def get_steps_poses_features_gaussian_convolution(img):
    """Convolve wavelet outcomes density with gaussian kernel."""
    sigma = (
        config.WAT_EMB_STP_SIGMA
        * config.WAT_RESOLUTION
        / (config.WAT_EMB_STP_LIM[1] - config.WAT_EMB_STP_LIM[0])
    )
    img_smooth = gaussian(img, sigma=sigma)
    img_smooth /= np.sum(img_smooth)
    return img_smooth


def get_umap_steps_poses_features_outcomes_densities(
    path_out_stps, path_kde, path_gau_kde
):
    """Probability density estimations for UMAP steps and poses outcomes."""
    out_stps = read_pickle(path_out_stps)
    axes_stps = [
        np.linspace(*config.WAT_EMB_STP_LIM, config.WAT_RESOLUTION + 1)
        for _ in range(2)
    ]
    kde_stps, _ = fastKDE.pdf(
        out_stps[:, 0],
        out_stps[:, 1],
        axes=axes_stps,
    )
    kde_stps = get_dynamic_range(kde_stps)
    write_pickle(kde_stps, path_kde)
    gau_kde_stps = get_steps_poses_features_gaussian_convolution(kde_stps)
    write_pickle(gau_kde_stps, path_gau_kde)


def get_steps_poses_features_watershed_segmentation(
    out_stps, img, path_labels, path_segmentation
):
    """Performs watershed segmentation over UMAP step and poses outcomes.

    img is a raw KDE, without gaussian smoothing.
    """

    print("1. Obtaining Gaussian KDE of the UMAP out_stps")
    img = get_steps_poses_features_gaussian_convolution(img)
    img = get_dynamic_range(img)
    plt.imshow(img, origin="lower")
    plt.savefig("gau_kde_stp_all.png", bbox_inches="tight", dpi=400)
    plt.close()

    print("2. Apply denoising median filter, with small window area")
    img = median(img, disk(config.WAT_RADIUS_DENOISE))
    img = get_dynamic_range(img)
    plt.imshow(img, origin="lower")

    print("3. Get image gradient to use as input for watershed segmentation")
    img_gradient = gradient(img, disk(config.WAT_RADIUS_GRADIENT))
    img_gradient = get_dynamic_range(img_gradient)
    plt.imshow(img_gradient, origin="lower")
    plt.savefig("gra_gau_kde_stp_all.png", bbox_inches="tight", dpi=400)
    plt.close()

    print("4. Compute watershed segmentation seeds, local maxima markers")
    img_cut = deepcopy(img)
    img_cut[img < config.WAT_CUT_THRESHOLD] = 0
    local_max_idx = peak_local_max(
        image=img_cut, footprint=disk(config.WAT_RADIUS_FOOTPRINT)
    )
    local_max = np.zeros_like(img, dtype=bool)
    local_max[tuple(local_max_idx.T)] = True
    plt.imshow(img_cut, origin="lower")
    plt.savefig("cut_gau_kde_stp_all.png", bbox_inches="tight", dpi=400)
    plt.close()
    del img_cut

    markers = ndi_label(local_max)[0]
    del local_max

    print("5. Compute compact watershed segmentation")
    predictions = watershed(image=img_gradient, markers=markers, compactness=1)
    print("Change labels")
    predictions = np.vectorize(config.LAB_STP_CHANGE_DICT.get)(predictions)
    segmentation = mark_boundaries(
        image=img,
        label_img=predictions,
        color=(0, 0, 0),
        outline_color=(1, 1, 1),
    )
    segmentation = rgb2gray(segmentation)
    write_pickle(segmentation, path_segmentation)
    out_stps_trans = deepcopy(out_stps)

    print("6. Normalize and scale UMAP out_stps to plot them over the image")
    out_stps_trans[:, 0] = (
        (out_stps[:, 0] - config.WAT_EMB_STP_LIM[0])
        / (config.WAT_EMB_STP_LIM[1] - config.WAT_EMB_STP_LIM[0])
        * config.WAT_RESOLUTION
    )
    out_stps_trans[:, 1] = (
        (out_stps[:, 1] - config.WAT_EMB_STP_LIM[0])
        / (config.WAT_EMB_STP_LIM[1] - config.WAT_EMB_STP_LIM[0])
        * config.WAT_RESOLUTION
    )

    print("7. Assign watershed segmentation labels to UMAP out_stps")
    labels = np.array(
        [predictions[int(outcome[1]), int(outcome[0])] for outcome in out_stps_trans],
        dtype=int,
    )
    del out_stps_trans
    write_pickle(labels, path_labels)
    print(f"Number of labels: {labels.max()}")
