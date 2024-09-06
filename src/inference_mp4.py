import argparse
import os
from glob import glob
from pathlib import Path

import cv2
from mmdet.apis import DetInferencer
from tqdm import tqdm
import torch
import random
import json

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet inferencer model")
    parser.add_argument("checkpoint", type=Path, help="checkpoint file")
    parser.add_argument(
        "device", type=int, default=0, help="device used for inference. `-1` means using cpu."
    )
    parser.add_argument(
        "--target_dir",
        "--target-dir",
        type=str,
        help="input directory of images to be predicted. It can be used wildcard.",
    )
    parser.add_argument(
        "--show_dir",
        "--show-dir",
        type=Path,
        help="directory where painted images will be saved. "
        "If specified, it will be automatically saved "
        "to the work_dir/timestamp/show_dir",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="The directory to save output prediction for offline evaluation",
    )
    parser.add_argument(
        "--pred_score_thr",
        "--pred-score-thr",
        type=float,
        default=0.3,
        help="Minimum score of bboxes to draw. Defaults to 0.3.",
    )
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    assert (
        args.show_dir is not None or args.out is not None
    ), "Either --show-dir or --out should be specified."
    
    video_paths = sorted(glob(f"{args.target_dir}/*.mp4", recursive=True))
    print(f"{args.target_dir} | {video_paths}")

    model = DetInferencer(
        weights=str(args.checkpoint),
        device=f"cuda:{args.device}" if args.device >= 0 else "cpu",
        show_progress=False,
    )
    for video_path in video_paths:
        # mp4 to images
        # inputの画像のフォルダ構成を保ち、frame番号をファイル名にする parent_dirが動画名、img_pathがframe番号に対応
        parent_dir = video_path.split("/")[-1].replace(".mp4", "")
        print(f"Processing {parent_dir}...")

        cap = cv2.VideoCapture(video_path)
                # --outの場合、APIで結果をjsonで保存するためパスを指定する
        if args.out is not None:
            out_dir = args.out / parent_dir
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = ""
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        batch_size = 1  # Subsample 2400 frames from each video
        sample_indices = random.sample(range(frame_count), 2400)  # Randomly sample 2400 frames
        sample_indices.sort()  # Sort the indices to maintain the order of frames
        frame_k = 0
        with tqdm(total=2400) as pbar:
            frames = []
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index in sample_indices:
                    frames.append(frame)
                else:
                    frame_index += 1
                    continue

                frame_index += 1

                # Process the batch when we reach the batch size
                if len(frames) == batch_size:
                    results = model(
                        inputs=frames,
                        return_vis=args.show_dir is not None,
                        no_save_vis=args.show_dir is None,
                        pred_score_thr=args.pred_score_thr,
                        no_save_pred=args.out is None,
                        #out_dir=out_dir,
                    )
                    frames = []  # Clear the batch after processing
                    pbar.update(batch_size)
                    results = results['predictions'][0]
                    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
                    with open(out_dir / "preds" / f"{frame_k}.json", "w") as f:
                        json.dump(results, f)

            # Process any remaining frames that didn't fill up a full batch
            if frames:
                results = model(
                    inputs=frames,
                    return_vis=args.show_dir is not None,
                    no_save_vis=args.show_dir is None,
                    pred_score_thr=args.pred_score_thr,
                    no_save_pred=args.out is None,
                    #out_dir=out_dir,
                )
                pbar.update(len(frames))
                frames = []  # Clear the batch after processing
                results = results['predictions'][0]
                (out_dir / "preds").mkdir(parents=True, exist_ok=True)
                with open(out_dir / "preds" / f"{frame_k}.json", "w") as f:
                    json.dump(results, f)

        cap.release()


if __name__ == "__main__":
    main()
