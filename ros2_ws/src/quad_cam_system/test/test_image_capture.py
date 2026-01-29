#!/usr/bin/env python3

import cv2
import os
from datetime import datetime
import time

def test_image_capture():
    """画像保存機能をテストする"""
    output_dir = '/tmp/captured_images'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 保存ディレクトリ: {output_dir}")
    print("🔍 利用可能なカメラを検索中...")
    
    # 利用可能なカメラを検索
    available_cameras = []
    for i in range(8):  # 最大8台まで検索
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append((i, cap, frame.shape))
                print(f"📷 カメラ{i} 検出: 解像度 {frame.shape[1]}x{frame.shape[0]}")
            else:
                cap.release()
        else:
            break
    
    if not available_cameras:
        print("❌ 利用可能なカメラが見つかりませんでした")
        return False
    
    print(f"✅ {len(available_cameras)}台のカメラが利用可能です")
    
    # 各カメラから画像を撮影して保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    captured_count = 0
    
    print("\n📸 画像撮影開始...")
    
    for cam_id, cap, shape in available_cameras:
        # 最大解像度を試行
        resolutions_to_try = [
            (1920, 1080),  # Full HD
            (1280, 720),   # HD
            (640, 480),    # VGA (fallback)
        ]
        
        best_resolution = None
        for width, height in resolutions_to_try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            ret, frame = cap.read()
            if ret and frame is not None:
                actual_width = frame.shape[1]
                actual_height = frame.shape[0]
                best_resolution = (actual_width, actual_height)
                break
        
        if best_resolution:
            # ファイル名を生成
            filename = f"test_capture_camera{cam_id}_{timestamp}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # 画像を保存
            success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            if success and os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"✅ カメラ{cam_id}: 保存成功")
                print(f"   📄 ファイル: {filepath}")
                print(f"   📐 解像度: {best_resolution[0]}x{best_resolution[1]}")
                print(f"   📦 サイズ: {file_size / 1024:.1f} KB")
                captured_count += 1
            else:
                print(f"❌ カメラ{cam_id}: 保存失敗")
        
        cap.release()
    
    print(f"\n📊 結果: {captured_count}/{len(available_cameras)}台のカメラから画像を保存しました")
    
    # 保存されたファイルを確認
    print("\n📋 保存されたファイル一覧:")
    try:
        files = os.listdir(output_dir)
        image_files = [f for f in files if f.endswith('.jpg')]
        if image_files:
            for f in image_files:
                filepath = os.path.join(output_dir, f)
                file_size = os.path.getsize(filepath)
                print(f"  📄 {f} ({file_size / 1024:.1f} KB)")
        else:
            print("  (画像ファイルなし)")
    except Exception as e:
        print(f"  エラー: {e}")
    
    return captured_count > 0

if __name__ == "__main__":
    print("🚀 画像保存テストを開始します...")
    success = test_image_capture()
    
    if success:
        print("\n✅ 画像保存テスト成功！")
        print("📁 保存場所: /tmp/captured_images")
    else:
        print("\n❌ 画像保存テスト失敗")
    
    print("\n🔍 ファイルを確認するには:")
    print("ls -la /tmp/captured_images/")