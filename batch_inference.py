import os
import argparse
import inference


def batch():
    parser = argparse.ArgumentParser(description="Wrapper script for COSeg Inference")
    parser.add_argument("--support", default="data/support", help="Path to support data")
    parser.add_argument("--query", default="working_dir/query", help="Path to query directory")

    parser.add_argument("--cfg", default="config/s3dis_COSeg_fs.yaml", help="Path to configuration file")
    parser.add_argument("--weight", default="data/weight/s31_1w5s.pth", help="Path to model weight file")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="Voxel size parameter, the lower the finer")
    args = parser.parse_args()

    batch_query_path = args.query

    if not os.path.exists(batch_query_path):
        raise ValueError(f"Query path {batch_query_path} does not exist.")

    count = len(os.listdir(batch_query_path))
    for i, query in enumerate(os.listdir(batch_query_path)):
        print(f"Processing batch inference: {query} ({i+1}/{count})")
        args.query = os.path.join(batch_query_path, query)
        args.vis_progress = False
        args.vis_result = False
        args.evaluate = False
        inference.main(args)
    print("Batch inference completed, total", count, "queries.")


if __name__ == "__main__":
    batch()
