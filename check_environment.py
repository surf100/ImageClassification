
import sys
import platform

import torch
import torchvision
import sklearn
import numpy as np
import pandas as pd
import matplotlib
import PIL
import tqdm


def main():
    print("=== Computer Vision Assignment 2 ===")
    print("Environment Check\n")

    print("Python:", sys.version.split()[0])
    print("Operating System:", platform.system())

    print("\n=== Library Versions ===")
    print("PyTorch:", torch.__version__)
    print("TorchVision:", torchvision.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("Pillow:", PIL.__version__)
    print("tqdm:", tqdm.__version__)

    print("\n=== GPU Check ===")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    print("\n=== Tensor Test ===")
    x = torch.rand(2, 3).to(device)
    print("Tensor:", x)
    print("Device:", x.device)

    print("\nEnvironment check completed successfully!")


if __name__ == "__main__":
    main()
