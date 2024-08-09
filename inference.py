import torch
import numpy as np
from model.coseg import COSeg
import open3d as o3d
from util import config
from util.data_util import data_prepare_v101_1 as data_prepare
from preprocess.room2blocks import room2blocks
import os
import time
import argparse


def ply_loader(file_path, fg_label=2, bg_label=12):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # XYZRGBL
    data = np.zeros((points.shape[0], 7))
    # XYZ
    data[:, 0:3] = points
    # set all colors to 200 (gray)
    data[:, 3:6] = 200
    # if the point is green, set the label
    is_green = np.all(colors == [0.0, 1.0, 0.0], axis=1)
    data[:, 6] = np.where(is_green, fg_label, bg_label)
    return data


def visualize_pcd(points, class_id, window_title="Point Cloud"):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])  # XYZ
    colors = points[:, 3:6] / 255.0  # RGB
    labels = points[:, 6]
    mask = (labels == class_id)
    colors[mask] = [0, 1, 0]
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_title, width=800, height=600)
    vis.add_geometry(point_cloud)
    vis.run()
    vis.destroy_window()


def process_pcd(pcd, class_id, voxel_size=0.02):
    coord = pcd[:, :3]
    feat = pcd[:, 3:6]
    label = pcd[:, 6]
    label[label != class_id] = 0
    label[label == class_id] = 1

    coord, feat, label, pcd_offset = data_prepare(coord, feat, label, voxel_size=voxel_size)
    x = torch.cat((coord, feat), dim=1)
    offset = torch.tensor([x.shape[0]], dtype=torch.int32)
    y = label.clone().detach()
    return x, offset, y, pcd_offset


def npy2blocks(data, room_name, block_size=1, stride=1, min_npts=1000):
    print(f"Pre-processing {room_name}")
    blocks_list = room2blocks(data, block_size=block_size, stride=stride, min_npts=min_npts)
    save_path = "data/query/blocks"

    if os.path.exists(save_path):
        file_list = os.listdir(save_path)
        for file in file_list:
            os.remove(os.path.join(save_path, file))
    else:
        os.makedirs(save_path)

    for i, block_data in enumerate(blocks_list):
        block_filename = room_name + "_block_" + str(i) + ".npy"
        np.save(os.path.join(save_path, block_filename), block_data)
    return save_path


class Evaluator:
    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.total_points = 0

    def update(self, predictions, ground_truth):
        predictions = predictions.clone().detach()
        ground_truth = ground_truth.clone().detach()

        tp = torch.sum((predictions == 1) & (ground_truth == 1)).item()
        tn = torch.sum((predictions == 0) & (ground_truth == 0)).item()
        fp = torch.sum((predictions == 1) & (ground_truth == 0)).item()
        fn = torch.sum((predictions == 0) & (ground_truth == 1)).item()

        self.TP += tp
        self.TN += tn
        self.FP += fp
        self.FN += fn
        self.total_points += len(ground_truth)

    def compute_metrics(self):
        accuracy = (self.TP + self.TN) / self.total_points if self.total_points != 0 else 0
        precision = self.TP / (self.TP + self.FP) if (self.TP + self.FP) != 0 else 0
        recall = self.TP / (self.TP + self.FN) if (self.TP + self.FN) != 0 else 0
        specificity = self.TN / (self.TN + self.FP) if (self.TN + self.FP) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        balanced_accuracy = (recall + specificity) / 2
        mcc_numerator = self.TP * self.TN - self.FP * self.FN
        mcc_denominator = ((self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN)) ** 0.5
        mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0
        iou = self.TP / (self.TP + self.FP + self.FN) if (self.TP + self.FP + self.FN) != 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'iou': iou,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'balanced_accuracy': balanced_accuracy,
            'mcc': mcc
        }


def main(args_in=None):
    parser = argparse.ArgumentParser("COSeg Inference")
    parser.add_argument("--support", default="data/support/3", help="Path to support data")
    parser.add_argument("--query", default="data/query/query.ply", help="Path to query data")

    parser.add_argument("--cfg", default="config/s3dis_COSeg_fs.yaml", help="Path to configuration file")
    parser.add_argument("--weight", default="data/weight/s31_1w5s.pth", help="Path to model weight file")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="Voxel size parameter, the lower the finer")

    parser.add_argument("--vis-progress", action="store_true", help="Visualize each block during inference")
    parser.add_argument("--vis-result", action="store_true", help="Visualize the final result")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the result and save metrics")

    if args_in is None:
        args = parser.parse_args()
    else:
        args = args_in

    ######################
    # init
    ######################
    support_dir = args.support
    query_file = args.query
    vis_progress = args.vis_progress
    vis_result = args.vis_result
    evaluate = args.evaluate
    voxel_size = args.voxel_size

    model_args = config.load_cfg_from_cfg_file(args.cfg)
    model_args.weight = args.weight

    model_args.train_gpu = [0]
    model_args.multiprocessing_distributed = False
    model_args.test = True
    model_args.eval_split = "test"
    model_args.k_shot = 5

    # model
    model = COSeg(model_args)
    model = model.cuda()
    checkpoint = torch.load(model_args.weight)
    pretrained_dict = checkpoint["state_dict"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict)
    torch.cuda.empty_cache()
    model.eval()

    ######################
    # support
    ######################

    class_id = 2
    support_files = []
    for filename in os.listdir(support_dir):
        if filename.endswith(".npy"):
            support_files.append(os.path.join(support_dir, filename))

    support_x, support_offset, support_y = None, None, None
    for i in range(0, model_args.k_shot):
        pcd_support = np.load(support_files[i])
        if vis_progress:
            visualize_pcd(pcd_support, class_id, window_title=f"Support {i + 1}")
        # load the first support
        if i == 0:
            support_x, support_offset, support_y, _ = process_pcd(pcd_support, class_id=class_id, voxel_size=voxel_size)
            continue
        # concatenate other supports
        support_x_temp, support_offset_temp, support_y_temp, _ = process_pcd(pcd_support, class_id=class_id,
                                                                             voxel_size=voxel_size)
        support_x = torch.cat((support_x, support_x_temp), dim=0)
        support_offset_temp = support_offset_temp + support_offset[-1]
        support_offset = torch.cat((support_offset, support_offset_temp), dim=0)
        support_y = torch.cat((support_y, support_y_temp), dim=0)

    ######################
    # query
    ######################
    time0 = time.time()

    if os.path.isdir(query_file):  # query_file is a directory containing blocks
        print("Using processed blocks as query")
        query_dir = query_file
    elif os.path.basename(query_file).endswith(".npy"):  # query_file is a npy file
        print("Using npy file as query")
        query_dir = npy2blocks(
            np.load(query_file),
            os.path.basename(query_file)[:-4]
        )
    elif os.path.basename(query_file).endswith(".ply"):  # query_file is a ply file
        print("Using ply file as query")
        query_dir = npy2blocks(
            ply_loader(query_file, fg_label=class_id),
            os.path.basename(query_file)[:-4]
        )
    else:
        raise FileNotFoundError(f"Query file error: {query_file}")

    query_blocks = []
    for filename in os.listdir(query_dir):
        if filename.endswith(".npy"):
            query_blocks.append(os.path.join(query_dir, filename))

    if len(query_blocks) == 0:
        raise FileNotFoundError(f"Error, no query blocks found")

    coords = []
    colors = []
    evaluator = None
    if evaluate:
        evaluator = Evaluator()

    for i, query_block in enumerate(query_blocks):
        # load and process query
        pcd_query = np.load(query_block)
        if vis_progress:
            visualize_pcd(pcd_query, class_id, window_title=f"Query {i + 1}")
        query_x, query_offset, query_y, pcd_offset = process_pcd(pcd_query, class_id=class_id, voxel_size=voxel_size)
        query_y = query_y.cuda(non_blocking=True)

        with torch.no_grad():
            output, loss = model(
                support_offset,
                support_x,
                support_y,
                query_offset,
                query_x,
                query_y,
                5)

        output = output.max(1)[1].squeeze(0)  # output: 1, c, pts

        if evaluate:
            evaluator.update(output, query_y)

        print(f"Processing {i + 1}/{len(query_blocks)} blocks")
        # append to point_clouds
        output = output.cpu().numpy()
        query_x = query_x.cpu().numpy()
        coord = query_x[:, :3]  # XYZ
        coord += pcd_offset
        color_map = {0: [0, 0, 0], 1: [0, 1, 0]}  # map：0->black，1->green
        color = np.array([color_map[value] for value in output])

        coords.extend(coord)
        colors.extend(color)

        if vis_progress:
            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(coord)
            pcd_pred.colors = o3d.utility.Vector3dVector(color)
            vis = o3d.visualization.Visualizer()
            vis.create_window(f"Prediction {i + 1}", width=800, height=600)
            vis.add_geometry(pcd_pred)
            vis.run()
            vis.destroy_window()

    ######################
    # save
    ######################
    filename_with_support_name = True

    save_dir = "working_dir/output"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.isdir(query_file):
        out_filename = os.path.join(save_dir, os.path.basename(query_file))
    else:
        out_filename = os.path.join(save_dir, os.path.basename(query_file)[:-4])

    if filename_with_support_name:
        out_filename += "_" + os.path.basename(support_dir)

    time1 = time.time()
    if evaluate:
        metrics = evaluator.compute_metrics()
        print_info = (f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
                      f"IoU: {metrics['iou']:.4f}, Recall: {metrics['recall']:.4f}, Specificity: "
                      f"{metrics['specificity']:.4f}, F1 Score: {metrics['f1_score']:.4f}, "
                      f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}, MCC: {metrics['mcc']:.4f}, "
                      f"Time: {time1 - time0:.2f}s")

        print(print_info)
        with open(out_filename + "_metrics.txt", "w") as f:
            f.write(print_info)
        print(f"Metrics saved to {out_filename}_metrics.txt")

    pcd_pred_whole = o3d.geometry.PointCloud()
    pcd_pred_whole.points = o3d.utility.Vector3dVector(coords)
    pcd_pred_whole.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_filename + "_result.ply", pcd_pred_whole)
    print(f"Output saved to {out_filename}_result.ply")

    if vis_result:
        o3d.visualization.draw_geometries([pcd_pred_whole])


if __name__ == "__main__":
    main()
