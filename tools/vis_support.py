import open3d as o3d
import numpy as np
import os
import sys


def create_text(text, position, scale=0.01, color=(0, 0, 0)):
    text_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=0.1).to_legacy()
    text_mesh.paint_uniform_color(color)
    text_mesh.transform([[scale, 0, 0, position[0]],
                         [0, scale, 0, position[1]],
                         [0, 0, scale, position[2]],
                         [0, 0, 0, 1]])

    return [text_mesh]


def load_point_clouds_from_directory(directory, fg_label=2):
    point_clouds = []
    files = sorted([f for f in os.listdir(directory) if f.endswith('.npy')])

    current_offset = 0.0

    for i, filename in enumerate(files):
        file_path = os.path.join(directory, filename)
        block_data = np.load(file_path)

        xyz = block_data[:, :3]
        rgb = block_data[:, 3:6] / 255.0
        l = block_data[:, 6]

        # move the point cloud to the origin
        center = np.mean(xyz, axis=0)
        xyz -= center

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # change the color of the points with fg_label to green
        colors = np.where(l[:, np.newaxis] == fg_label, [0, 1, 0], rgb)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # move the point cloud to the right
        pcd.translate([current_offset, 0, 0])

        x_range = np.max(xyz[:, 0]) - np.min(xyz[:, 0])
        current_offset += x_range + 2

        # create text and move it to the top of the point cloud
        max_z = np.max(xyz[:, 2])
        text_position = np.array([current_offset - x_range - 2,
                                  0,
                                  max_z + 0.1])

        text_meshes = create_text(filename, text_position, scale=0.005, color=(0, 0, 0))

        point_clouds.append(pcd)
        point_clouds.extend(text_meshes)

    return point_clouds


def main():
    if len(sys.argv) != 2:
        print("Usage: python vis_support.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    point_clouds = load_point_clouds_from_directory(directory)
    o3d.visualization.draw_geometries(point_clouds, mesh_show_back_face=True)


if __name__ == "__main__":
    main()
