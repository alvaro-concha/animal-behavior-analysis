import numba
import numpy as np
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt


def _get_embedding(umap_object):
    if hasattr(umap_object, "embedding_"):
        return umap_object.embedding_
    elif hasattr(umap_object, "embedding"):
        return umap_object.embedding
    else:
        raise ValueError("Could not find embedding attribute of umap_object")


def _get_extent(points):
    """Compute bounds on a space with appropriate padding"""
    min_x = np.nanmin(points[:, 0])
    max_x = np.nanmax(points[:, 0])
    min_y = np.nanmin(points[:, 1])
    max_y = np.nanmax(points[:, 1])

    extent = (
        np.round(min_x - 0.05 * (max_x - min_x)),
        np.round(max_x + 0.05 * (max_x - min_x)),
        np.round(min_y - 0.05 * (max_y - min_y)),
        np.round(max_y + 0.05 * (max_y - min_y)),
    )

    return extent


@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _red(x):
    return (x & 0xFF0000) >> 16


@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _green(x):
    return (x & 0x00FF00) >> 8


@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _blue(x):
    return x & 0x0000FF


def _embed_datashader_in_an_axis(datashader_image, ax, extent):
    img_rev = datashader_image.data[::-1]
    mpl_img = np.dstack([_blue(img_rev), _green(img_rev), _red(img_rev)])
    ax.imshow(mpl_img, extent=extent)
    return ax


def hammer_connectivity(
    edges,
    extent,
    dpi,
    edge_cmap="gray_r",
    background="white",
    width=800,
    height=800,
):
    """Plot connectivity relationships of the underlying UMAP
    simplicial set data structure. Internally UMAP will make
    use of what can be viewed as a weighted graph. This graph
    can be plotted using the layout provided by UMAP as a
    potential diagnostic view of the embedding. Currently this only works
    for 2D embeddings.
    """
    canvas = ds.Canvas(
        plot_width=width,
        plot_height=height,
        x_range=(extent[0], extent[1]),
        y_range=(extent[2], extent[3]),
    )

    edge_img = tf.shade(
        canvas.line(edges, "x", "y", agg=ds.sum("weight")),
        cmap=plt.get_cmap(edge_cmap),
        how="eq_hist",
        alpha=255 * 0.9,
    )
    edge_img = tf.set_background(edge_img, background)

    fig = plt.figure(figsize=(width / dpi, height / dpi))
    ax = fig.add_subplot(111)

    _embed_datashader_in_an_axis(edge_img, ax, extent)

    return ax
