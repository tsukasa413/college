"""
リアルタイムでUSBカメラから画像を取得するためのモジュール
"""
import cv2
import numpy as np
import torch

class MultiCameraCapture:
    def __init__(self, camera_indices=[0,1,2,3], width=1216, height=1216):
        """
        複数のカメラを初期化
        Args:
            camera_indices: カメラのインデックスリスト（/dev/videoXのX）
            width: キャプチャ画像の幅
            height: キャプチャ画像の高さ
        """
        self.caps = []
        self.camera_indices = camera_indices
        
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                raise RuntimeError(f"カメラ {idx} をオープンできません")
            
            # カメラの解像度を設定
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.caps.append(cap)

    def capture_images(self, matching_resolution=(1024, 1024), rgb_resolution=(1216, 1216)):
        """
        全カメラから同時に画像を取得
        Args:
            matching_resolution: マッチング用の解像度 (width, height)
            rgb_resolution: RGB出力用の解像度 (width, height)
        Returns:
            images_to_match: マッチング用の低解像度画像リスト
            images_to_stitch: スティッチング用の高解像度画像リスト
        """
        images_to_match = []
        images_to_stitch = []
        
        for cap in self.caps:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # マッチング用の低解像度画像
            frame_match = cv2.resize(frame, matching_resolution)
            images_to_match.append(frame_match)
            
            # スティッチング用の高解像度画像
            frame_stitch = cv2.resize(frame, rgb_resolution)
            images_to_stitch.append(frame_stitch)
            
        return images_to_match, images_to_stitch

    def release(self):
        """カメラリソースの解放"""
        for cap in self.caps:
            cap.release()