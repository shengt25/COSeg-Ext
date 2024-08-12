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
import subprocess
import re


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


def ply_loader(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # XYZRGBL
    data = np.zeros((points.shape[0], 7))
    data[:, 0:3] = points
    data[:, 3:6] = colors * 255
    data[:, 6] = -1  # set all labels to -1
    return data


def visualize_pcd(points, window_title="Point Cloud", class_id=None):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])  # XYZ
    if class_id is not None:
        labels = points[:, 6]
        mask = (labels == class_id)
        colors = np.zeros_like(points[:, 3:6])
        colors[mask] = [0, 1, 0]
    else:
        colors = points[:, 3:6] / 255.0  # RGB

    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_title)
    vis.add_geometry(point_cloud)
    vis.run()
    vis.destroy_window()


def process_pcd(pcd, class_id, voxel_size=0.02):
    coord = pcd[:, :3]
    feat = pcd[:, 3:6]  # color with range [0, 255], data_prepare will normalize it to [0, 1]
    label = pcd[:, 6]
    # note: modifying coord, feat, label will change the original pcd,
    #       in python it is passed by reference, not by value

    label[label != class_id] = 0
    label[label == class_id] = 1

    coord, feat, label, pcd_offset = data_prepare(coord, feat, label, voxel_size=voxel_size)
    x = torch.cat((coord, feat), dim=1)
    offset = torch.tensor([x.shape[0]], dtype=torch.int32)
    y = label.clone().detach()
    return x, offset, y, pcd_offset


def npy2blocks(data, room_name, save_path, block_size=1, stride=1, min_npts=1000):
    print(f"Pre-processing {room_name}")
    blocks_list = room2blocks(data, block_size=block_size, stride=stride, min_npts=min_npts)

    if os.path.exists(save_path):
        file_list = os.listdir(save_path)
        for file in file_list:
            os.remove(os.path.join(save_path, file))
    else:
        os.makedirs(save_path)

    for i, block_data in enumerate(blocks_list):
        block_filename = room_name + "_block_" + str(i) + ".npy"
        np.save(os.path.join(save_path, block_filename), block_data)


def get_vram_usage():
    # get the process ID of the current process
    pid = os.getpid()

    command = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
    output = subprocess.check_output(command, shell=True).decode('utf-8')

    for line in output.splitlines():
        if str(pid) in line:
            match = re.search(r'(\d+)\s+MiB', line)
            if match:
                used_memory = int(match.group(1))
            else:
                return -1
            break
    else:
        return -1
    return used_memory


def main(args_in=None):
    ######################
    # settings
    ######################
    pred_save_dir = "working_dir/output"
    if not os.path.exists(pred_save_dir):
        os.makedirs(pred_save_dir)

    query_blocks_dir_default = "data/query/blocks"

    class_id = 2
    ######################
    # args
    ######################

    parser = argparse.ArgumentParser("COSeg Inference")
    parser.add_argument("query", default="working_dir/input", help="Path to query data")

    parser.add_argument("--support", default="data/support", help="Path to support data")
    parser.add_argument("--cfg", default="config/s3dis_COSeg_fs.yaml", help="Path to configuration file")
    parser.add_argument("--weight", default="data/weight/s31_1w5s.pth", help="Path to model weight file")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="Voxel size parameter, the lower the finer")

    parser.add_argument("--evaluate", action="store_true", help="Evaluate the result and save metrics")
    parser.add_argument("--vis-progress", action="store_true", help="Visualize each block during inference")
    parser.add_argument("--vis-result", action="store_true", help="Visualize the final result")

    if args_in is None:
        args = parser.parse_args()
    else:
        args = args_in

    ######################
    # init model
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

    args_str = ', '.join([f"{arg}: {getattr(args, arg)}" for arg in vars(args)])
    print("Options:", args_str)

    support_files = []
    for filename in os.listdir(support_dir):
        if filename.endswith(".npy"):
            support_files.append(os.path.join(support_dir, filename))

    support_x, support_offset, support_y = None, None, None
    if len(support_files) == 0:
        print(f"Error, no support data found in {support_dir}")
        return
    for i in range(0, model_args.k_shot):
        pcd_support = np.load(support_files[i])
        if vis_progress:
            visualize_pcd(pcd_support, f"Support data: {os.path.basename(support_files[i])}")
            visualize_pcd(pcd_support, f"Support label: {os.path.basename(support_files[i])}", class_id)
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
    query_type_ply = False

    query_blocks_dir = query_blocks_dir_default
    if not os.path.exists(query_blocks_dir):
        os.makedirs(query_blocks_dir)

    if not os.path.exists(query_file):
        print(f"Error, query file {query_file} does not exist")
        return
    elif os.path.isdir(query_file):  # query_file is a directory containing npy blocks
        print("Using processed blocks as query")
        query_blocks_dir = query_file
    elif os.path.basename(query_file).endswith(".npy"):  # query_file is a npy file, convert to blocks
        print("Using npy file as query")
        npy2blocks(np.load(query_file),
                   os.path.basename(query_file)[:-4],
                   query_blocks_dir)
    elif os.path.basename(query_file).endswith(".ply"):  # query_file is a ply file, convert to npy and then blocks
        print("Using ply file as query")
        npy2blocks(ply_loader(query_file),
                   os.path.basename(query_file)[:-4],
                   query_blocks_dir)
        query_type_ply = True
    else:
        print(f"Unsupported query file : {query_file}")
        return

    query_blocks = []
    for filename in os.listdir(query_blocks_dir):
        if filename.endswith(".npy"):
            query_blocks.append(os.path.join(query_blocks_dir, filename))

    if len(query_blocks) == 0:
        print(f"Error, no query blocks found")
        return

    evaluator = None
    max_vram_usage = None
    if evaluate:
        max_vram_usage = -1
        if query_type_ply:
            print("Using ply file, only time, points count and vram usage will be recorded")
        else:
            evaluator = Evaluator()

    print("")  # newline

    coords = []
    colors = []
    points_count = 0
    for i, query_block in enumerate(query_blocks):
        # load and process query
        pcd_query = np.load(query_block)
        if evaluate:
            # points before processing, because process_pcd will change the points (pass by reference)
            points_count += pcd_query.shape[0]
        if vis_progress:
            visualize_pcd(pcd_query, f"Query data: {os.path.basename(query_block)}")
            if not query_type_ply:
                visualize_pcd(pcd_query, f"Query label: {os.path.basename(query_block)}", class_id)
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
            max_vram_usage = max(max_vram_usage, get_vram_usage())
            if evaluator is not None:
                evaluator.update(output, query_y)

        progress_str = f"Processing {os.path.basename(query_file)}: {(i + 1) * 100 / len(query_blocks):.2f}% ({i + 1}/{len(query_blocks)} blocks)"
        print(f"\r{' ' * len(progress_str)}", end='\r')  # clear line
        print(progress_str, end='\r')

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
            vis.create_window(f"Prediction {os.path.basename(query_block)}")
            vis.add_geometry(pcd_pred)
            vis.run()
            vis.destroy_window()

    ######################
    # save
    ######################

    # clean up temp query blocks
    if os.path.exists(query_blocks_dir_default):
        file_list = os.listdir(query_blocks_dir_default)
        for file in file_list:
            os.remove(os.path.join(query_blocks_dir_default, file))

    if os.path.isdir(query_file):
        pred_base_filename = os.path.join(pred_save_dir, os.path.basename(query_file))
    else:
        pred_base_filename = os.path.join(pred_save_dir, os.path.basename(query_file)[:-4])

    time1 = time.time()
    if evaluate:
        formatted_points_count = f"{points_count:_}".replace('_', ' ')

        if evaluator is not None:
            metrics = evaluator.compute_metrics()

            print_info = (f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
                          f"IoU: {metrics['iou']:.4f}, Recall: {metrics['recall']:.4f}, Specificity: "
                          f"{metrics['specificity']:.4f}, F1 Score: {metrics['f1_score']:.4f}, "
                          f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}, MCC: {metrics['mcc']:.4f}, "
                          f"Time: {time1 - time0:.2f}s, Points Count: {formatted_points_count},"
                          f" Max VRAM Usage: {max_vram_usage} MB")
        else:
            print_info = (f"Time: {time1 - time0:.2f}s, Points Count: {formatted_points_count}, "
                          f"Max VRAM Usage: {max_vram_usage} MiB")

        print(print_info)
        pred_metrics_filename = pred_base_filename + "_metrics.txt"
        if os.path.exists(pred_metrics_filename):
            i = 1
            while os.path.exists(pred_metrics_filename):
                pred_metrics_filename = pred_base_filename + f"_metrics_{i}.txt"
                i += 1
        with open(pred_metrics_filename, "w") as f:
            f.write(print_info)

        print(f"Metrics saved to {pred_metrics_filename}")

    pcd_pred_whole = o3d.geometry.PointCloud()
    pcd_pred_whole.points = o3d.utility.Vector3dVector(coords)
    pcd_pred_whole.colors = o3d.utility.Vector3dVector(colors)

    pred_ply_filename = pred_base_filename + "_result.ply"
    if os.path.exists(pred_ply_filename):
        i = 1
        while os.path.exists(pred_ply_filename):
            pred_ply_filename = pred_base_filename + f"_result_{i}.ply"
            i += 1

    o3d.io.write_point_cloud(pred_ply_filename, pcd_pred_whole)
    print(f"Output saved to {pred_ply_filename}")

    if vis_result:
        o3d.visualization.draw_geometries([pcd_pred_whole])


if __name__ == "__main__":
    main()
