import open3d as o3d
import numpy as np
import random
import os
import sys


def create_text(text, position, scale=0.01, color=(0.4, 0.1, 0.9)):
    text_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=0.1).to_legacy()
    text_mesh.paint_uniform_color(color)
    text_mesh.transform([[scale, 0, 0, position[0]],
                         [0, scale, 0, position[1]],
                         [0, 0, scale, position[2]],
                         [0, 0, 0, 1]])

    return [text_mesh]


def load_pcd(directory, fg_label=None):
    point_clouds = []
    files = sorted([f for f in os.listdir(directory) if f.endswith('.npy')])

    for i, filename in enumerate(files):
        file_path = os.path.join(directory, filename)
        block_data = np.load(file_path)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(block_data[:, :3])

        if fg_label is not None:
            color = [random.uniform(0, 1), 0, random.uniform(0, 1)]
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (block_data.shape[0], 1)))
            mask = block_data[:, 6] == fg_label
            pcd.colors = o3d.utility.Vector3dVector(np.where(mask[:, np.newaxis], [0, 1, 0], pcd.colors))
        else:
            color = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (block_data.shape[0], 1)))

        max_z = np.max(block_data[:, 2])
        center = np.mean(block_data[:, :3], axis=0)
        text_position = center.copy()
        text_position[2] = max_z + 0.1

        text_meshes = create_text(filename[-12:-4], text_position, scale=0.005, color=(1, 1, 1))

        point_clouds.append(pcd)
        point_clouds.extend(text_meshes)

    return point_clouds


def main():
    fg_label = None

    if len(sys.argv) == 2:
        directory = sys.argv[1]
    elif len(sys.argv) == 3:
        directory = sys.argv[1]
        fg_label = int(sys.argv[2])
    else:
        print("Usage: python vis_blocks.py <directory> [fg_label]")
        sys.exit(1)

    point_clouds = load_pcd(directory, fg_label)
    o3d.visualization.draw_geometries(point_clouds, mesh_show_back_face=True)


if __name__ == "__main__":
    main()
