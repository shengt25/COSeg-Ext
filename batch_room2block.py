import os
import numpy as np
import sys
from inference import npy2blocks


def main():
    if len(sys.argv) != 2:
        print("Usage: python batch_room2block.py <file_dir>")
        sys.exit(1)

    file_dir = sys.argv[1]
    file_names = os.listdir(file_dir)
    for file_name in file_names:
        if not file_name.endswith(".npy"):
            continue

        data = np.load(os.path.join(file_dir, file_name))
        save_path = "working_dir/source/" + os.path.basename(file_name)[:-4]
        npy2blocks(data, os.path.basename(file_name)[:-4], save_path,
                   block_size=1, stride=1, min_npts=1000)
        print(f"Saved block data to {save_path}")


if __name__ == "__main__":
    main()
