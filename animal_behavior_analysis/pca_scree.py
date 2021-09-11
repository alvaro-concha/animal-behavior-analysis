"""Check IncrementalPCA scree plot."""
from utilities import read_pickle
import config_dodo
import numpy as np
import plotly.graph_objects as go

target_pickle_path = config_dodo.WAV_PATH / "pca_fit_wav.pickle"
pca = read_pickle(target_pickle_path)
scree = np.cumsum(pca.explained_variance_ratio_)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.arange(len(scree) + 1),
        y=np.concatenate(([0.0], scree)),
        mode="lines+markers",
    )
)
fig.update_xaxes(type="log")
fig.show()
