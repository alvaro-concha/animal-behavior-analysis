"""pathlib testing"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle5
import h5py


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.absolute()

    filename = repo_path / "Figures/test.png"
    x = np.linspace(0, 2 * np.pi, 1000)
    y = np.sin(x)
    plt.plot(x, y)
    plt.savefig(filename, bbox_inches="tight")

    obj = {"x": x, "y": y}
    pickle_path = repo_path / "Data/test.pickle"
    with open(pickle_path, "wb") as file:
        pickle5.dump(obj, file, protocol=-1, fix_imports=False)

    with h5py.File("mytestfile.hdf5", "w") as f:
        dset = f.create_dataset("mydataset", (100,), dtype="i")
