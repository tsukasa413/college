"""
リアルタイムカメラ入力からの3D再構成を実行
"""
import cupy
from depth_estimation import RGBD_Estimator
from utils import parse_json_calib
from camera_capture import MultiCameraCapture

import os.path 
import torch
import json
import argparse
import cv2
import numpy as np
import signal
import sys

def signal_handler(sig, frame):
    print('Ctrl+C が押されました。終了します。')
    if 'camera' in globals():
        camera.release()
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="resources/calibration.json",
                      help="カメラキャリブレーションファイルのパス")
    parser.add_argument('--references_indices', nargs="*", type=int, default=[0, 2])
    parser.add_argument('--min_dist', type=float, default=0.55)
    parser.add_argument('--max_dist', type=float, default=100)
    parser.add_argument('--candidate_count', type=int, default=32)
    parser.add_argument('--sigma_i', type=float, default=10)
    parser.add_argument('--sigma_s', type=float, default=25)
    parser.add_argument('--matching_resolution', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('--rgb_to_stitch_resolution', nargs=2, type=int, default=[1216, 1216])
    parser.add_argument('--panorama_resolution', nargs=2, type=int, default=[2048, 1024])
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()

    # カメラキャリブレーション読み込み
    with open(args.config_path) as f:
        raw_calibration = json.load(f)['value0']
    calibrations = parse_json_calib(raw_calibration, args.matching_resolution, args.device)

    # リファレンスビューポイントを設定
    reprojection_viewpoint = torch.zeros([3], device=args.device)
    for references_index in args.references_indices:
        reprojection_viewpoint += calibrations[references_index].rt[:3, 3] / len(args.references_indices)

    # マスクの設定（全画素有効）
    masks = [torch.ones(args.matching_resolution, device=args.device).unsqueeze(0) 
            for _ in range(len(calibrations))]

    # RGBDエスティメータの初期化
    rgbd_estimator = RGBD_Estimator(calibrations, args.min_dist, args.max_dist, args.candidate_count,
                                   args.references_indices, reprojection_viewpoint, masks,
                                   args.matching_resolution, args.rgb_to_stitch_resolution, 
                                   args.panorama_resolution, args.sigma_i, args.sigma_s, args.device)

    # カメラの初期化
    camera = MultiCameraCapture(camera_indices=[0,1,2,3],
                              width=args.rgb_to_stitch_resolution[0],
                              height=args.rgb_to_stitch_resolution[1])

    try:
        while True:
            # カメラから画像を取得
            fisheye_images, reference_fisheye_images = camera.capture_images(
                tuple(args.matching_resolution),
                tuple(args.rgb_to_stitch_resolution))

            if len(fisheye_images) == 4 and len(reference_fisheye_images) == 4:
                # GPU用のテンソルに変換
                fisheye_images = [torch.tensor(img, device=args.device) for img in fisheye_images]
                reference_fisheye_images = [torch.tensor(img, device=args.device) 
                                          for img in reference_fisheye_images]

                # RGB-Dパノラマの推定
                rgb, distance = rgbd_estimator.estimate_RGBD_panorama(
                    fisheye_images, reference_fisheye_images)

                # 距離マップの可視化
                distance_map = 1 / distance.cpu().numpy()
                distance_map = ((distance_map - 1 / args.max_dist) 
                              / (1 / args.min_dist - 1 / args.max_dist))
                distance_map = np.clip(255 * distance_map, 0, 255).astype(np.uint8)
                distance_map = cv2.applyColorMap(distance_map, cv2.COLORMAP_MAGMA)

                # 結果の表示
                cv2.imshow("RGB Panorama", rgb.cpu().numpy())
                cv2.imshow("Distance Map", distance_map)

            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()