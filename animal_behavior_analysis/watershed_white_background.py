import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from general import try_create_folder, write_pickle, read_pickle
from fastkde import fastKDE
from skimage.util import img_as_ubyte
from skimage.filters import gaussian
from skimage.morphology import disk
from skimage.filters.rank import median, gradient
from skimage.feature import peak_local_max
from scipy.ndimage import label as ndi_label
from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2gray
from umap import UMAP

############################## GLOBAL SETUP ################################

sns.set_context("paper", font_scale=1.5)
plt.rc("figure", figsize=(5, 7))
plt.rc("axes", labelsize=16, labelcolor="k")
plt.rc("xtick", labelsize=14, color="k")
plt.rc("ytick", labelsize=14, color="k")
plt.rc("legend", fontsize=14)
plt.rc("legend", title_fontsize=14)
plt.rc("savefig", bbox="tight")

################################ FUNCTIONS #################################


def create_plot(outcomes, resolution):
    """
    Creates an empty plot, returns figure and axes
    Parameters
    ----------
    outcomes: array_like
            UMAP outcomes used only to set extreme tick values
    resolution: int
            Used to correctly position the ticks of the plot at its extremes
    """
    fig = plt.figure()
    ax = plt.gca()
    z1_min_round = np.round(outcomes[:, 0].min(), decimals=0)
    z1_max_round = np.round(outcomes[:, 0].max(), decimals=0)
    z2_min_round = np.round(outcomes[:, 1].min(), decimals=0)
    z2_max_round = np.round(outcomes[:, 1].max(), decimals=0)
    plt.xticks([5 * resolution // 128, 123 * resolution // 128])
    ax.set_xticklabels([rf"${z1_min_round:.0f}$", f"{z1_max_round:.0f}"])
    plt.yticks([5 * resolution // 128, 123 * resolution // 128])
    ax.set_yticklabels([rf"${z2_min_round:.0f}$", f"{z2_max_round:.0f}"])
    plt.xlabel(r"UMAP$_1$")
    plt.ylabel(r"UMAP$_2$")

    return fig, ax


def dynamic_range(img):
    """
    Returns image with full dynamic range
    Parameters
    ----------
    img: array_like
            Image whose intensities are to be rescaled to full dynamic range
    """
    img = (img - img.min()) / (img.max() - img.min())
    img = img_as_ubyte(img)
    return img


def umap_gaussian_kde(
    outcomes, resolution=2048, epsilon=0.1, sig_umap=1.0, folder="Figures/Watershed"
):
    """
    Plots UMAP image PDF. Returns PDF img
    Parameters
    ----------
    outcomes: array_like
            UMAP outcomes used to estimate their probability density
    resolution: int
            Resolution for fastKDE probability density estimation
    epsilon: float
            Axis expansion factor for fastKDE
    sig_umap: float
            Standard deviation for the probability density estimation
    folder: str
            Name of the figure folder to save the Watershed segmentation in
    """
    img, _ = fastKDE.pdf(
        outcomes[:, 0],
        outcomes[:, 1],
        numPoints=resolution + 1,
        axisExpansionFactor=epsilon,
    )
    img = dynamic_range(img)

    z1_min = outcomes[:, 0].min()
    z1_max = outcomes[:, 0].max()
    sigma = sig_umap * resolution * (1 - epsilon) / (z1_max - z1_min)

    img = gaussian(img, sigma=sigma)
    img /= np.sum(img)

    fig, ax = create_plot(outcomes, resolution)
    pdf = plt.imshow(img, origin="lower", cmap="YlOrRd")

    img_max = img.max()
    exponent = np.floor(np.log10(img_max))
    significant = img_max * 10 ** (-exponent)

    cbar = fig.colorbar(
        pdf,
        ax=ax,
        label="Gaussian KDE",
        pad=0.15,
        orientation="horizontal",
        ticks=[img.min(), img.max()],
    )
    cbar.ax.tick_params(size=0)
    cbar.set_ticklabels(
        ["0", f"{significant:.1f}" + r"$\times 10^{" + f"{exponent:.0f}" + r"}$"]
    )
    plt.savefig(folder + "/" + "umap_outcomes_kde.pdf")
    plt.close()

    return img


def annotate_labels(ax, label_nums, labels, outcomes):
    """
    Annotates labels to a UMAP segmented embedding
    Parameters
    ----------
    ax: plt.axes
            Axes to annotate labels on
    label_nums: array_like
            Unique label values
    labels: array_like
            Labels associated to the UMAP outcomes
    outcomes: array_like
            UMAP outcomes, the annotations will be displayed in their centroids
    """
    for i in label_nums:
        label_median = np.median(outcomes[labels == i], axis=0)
        label_txt = ax.text(
            label_median[0],
            label_median[1],
            str(i),
            fontsize=16,
            color="k",
            horizontalalignment="center",
            verticalalignment="center",
        )
        label_txt.set_path_effects(
            [
                PathEffects.Stroke(
                    linewidth=1,
                    foreground="w",
                ),
                PathEffects.Normal(),
            ]
        )


def run_watershed(
    resolution=2048,
    epsilon=0.1,
    sig_umap=1.0,
    radius_denoise=2,
    radius_gradient=2,
    radius_footprint=30,
    cut_threshold=20,
    folder="Figures/Watershed",
):
    """
    Performs Compact Watershed Transform over UMAP outcomes
    Parameters
    ----------
    resolution: int
            Resolution for fastKDE probability density estimation
    epsilon: float
            Axis expansion factor for fastKDE
    sig_umap: float
            Standard deviation for the probability density estimation
    radius_denoise: int
            Radius for the disk-shaped median denoising filter mask
    radius_gradient: int
            Radius for the disk-shaped gradient filter mask
    radius_footprint: int
            Radius for the disk-shaped mask to search local maxima within
    cut_threshold: float
            8-bit intensity values below these threshold are set to 0
    folder: str
            Name of the figure folder to save the Watershed segmentation in
    """
    try_create_folder(folder)
    outcomes = read_pickle("umap_outcomes.pickle", folder="Data").embedding_

    print("1. Obtaining Gaussian KDE of the UMAP outcomes")
    img = umap_gaussian_kde(
        outcomes, resolution=2048, epsilon=0.1, sig_umap=sig_umap, folder=folder
    )
    img = dynamic_range(img)

    print("2. Apply denoising median filter, with small window area")
    img = median(img, disk(radius_denoise))
    img = dynamic_range(img)

    print("3. Get image gradient to use as input for watershed segmentation")
    img_gradient = gradient(img, disk(radius_gradient))
    img_gradient = dynamic_range(img_gradient)

    print("4. Compute watershed segmentation seeds, local maxima markers")
    img_cut = np.copy(img)
    img_cut[img < cut_threshold] = 0
    local_max = peak_local_max(
        image=img_cut, footprint=disk(radius_footprint), indices=False
    )
    del img_cut

    markers = ndi_label(local_max)[0]
    del local_max

    print("5. Compute compact watershed segmentation")
    predictions = watershed(image=img_gradient, markers=markers, compactness=1)

    fig, ax = create_plot(outcomes, resolution)
    z1_min = outcomes[:, 0].min()
    z1_max = outcomes[:, 0].max()
    z2_min = outcomes[:, 1].min()
    z2_max = outcomes[:, 1].max()
    outcomes_trans = copy.deepcopy(outcomes)

    print("6. Normalize and scale UMAP outcomes to plot them over the image")
    outcomes_trans[:, 0] = (outcomes[:, 0] - z1_min) / (z1_max - z1_min) * resolution
    outcomes_trans[:, 1] = (outcomes[:, 1] - z2_min) / (z2_max - z2_min) * resolution

    outcomes_trans -= [resolution * 0.5, resolution * 0.5]
    outcomes_trans /= 1 + epsilon
    outcomes_trans += [resolution * 0.5, resolution * 0.5]

    print("7. Assign watershed segmentation labels to UMAP outcomes")
    labels = np.array(
        [predictions[int(outcome[1]), int(outcome[0])] for outcome in outcomes_trans],
        dtype=int,
    )
    write_pickle(labels, "watershed_labels.pickle", folder="Data")
    label_nums = np.unique(labels)

    segmentation = mark_boundaries(
        image=img,
        label_img=predictions,
        # color=(1, 1, 1),
        color=(0, 0, 0),
        outline_color=(1, 1, 1),
    )
    segmentation = rgb2gray(segmentation)

    plt.imshow(segmentation, origin="lower", cmap="binary")
    scat = plt.scatter(
        outcomes_trans[:, 0],
        outcomes_trans[:, 1],
        s=0.1,
        alpha=0.1,
        c=labels,
        cmap="gist_rainbow",
    )
    annotate_labels(ax, label_nums, labels, outcomes_trans)
    sns.despine(trim=True)
    plt.savefig(folder + "/" + "watershed_segmentation.png", dpi=600)
    plt.close()


if __name__ == "__main__":

    run_watershed(
        resolution=2048,
        epsilon=0.1,
        sig_umap=2.0,  # 1., 1.5, 2.
        radius_denoise=2,
        radius_gradient=10,  # 2, 10
        radius_footprint=30,
        cut_threshold=20,
        folder="Figures/Watershed",
    )
