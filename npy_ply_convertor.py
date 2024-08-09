import numpy as np
import open3d as o3d
import os


def npy2ply(npy_file_path, ply_file_path=None, fg_label=2):
    if not npy_file_path.endswith(".npy"):
        return

    data = np.load(npy_file_path)

    # Nx7: XYZRGBL
    xyz = data[:, :3]
    rgb = data[:, 3:6].astype(np.uint8)
    labels = data[:, 6]

    # Set label 2 to green
    rgb[labels == fg_label] = [0, 255, 0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

    if ply_file_path is None:
        ply_file_path = npy_file_path[:-4] + ".ply"

    o3d.io.write_point_cloud(ply_file_path, pcd)
    print(f"Converted {npy_file_path} to {ply_file_path}")


def ply2npy(ply_file_path, npy_file_path=None, fg_label=2, bg_label=12):
    if not ply_file_path.endswith(".ply"):
        return
    pcd = o3d.io.read_point_cloud(ply_file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Nx7: XYZRGBL
    npy_data = np.zeros((points.shape[0], 7))
    npy_data[:, 0:3] = points

    # set all colors to 200 (gray)
    npy_data[:, 3:6] = 200

    # if the point is green, set the label
    is_green = np.all(colors == [0.0, 1.0, 0.0], axis=1)
    npy_data[:, 6] = np.where(is_green, fg_label, bg_label)

    if npy_file_path is None:
        npy_file_path = ply_file_path[:-4] + ".npy"
    np.save(npy_file_path, npy_data)
    print(f"Converted {ply_file_path} to {npy_file_path}")


if __name__ == "__main__":
    file_dir = "data/support5/12"
    for file in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file)
        ply2npy(file_path)
        # npy2ply(file_path)
