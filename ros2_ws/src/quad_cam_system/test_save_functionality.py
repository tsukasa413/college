#!/usr/bin/env python3

import cv2
import numpy as np
import os
from datetime import datetime
import time

def test_image_save_functionality():
    """画像保存機能をテスト（仮想画像を使用）"""
    output_dir = '/tmp/captured_images'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 保存ディレクトリ: {output_dir}")
    print("🎨 仮想画像を生成して保存機能をテスト中...")
    
    # 異なる解像度の仮想画像を生成
    test_resolutions = [
        (1920, 1080),  # Full HD
        (1944, 1096),  # Max resolution
        (1280, 720),   # HD
        (640, 480),    # VGA
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    captured_count = 0
    
    print("\n📸 画像保存テスト開始...")
    
    for cam_id, (width, height) in enumerate(test_resolutions):
        # カラフルな仮想画像を生成
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 背景グラデーション
        for y in range(height):
            for x in range(width):
                image[y, x] = [
                    int(255 * x / width),           # Red gradient
                    int(255 * y / height),          # Green gradient
                    int(255 * (1 - x / width))      # Blue gradient
                ]
        
        # テキストを追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_lines = [
            f"Camera {cam_id}",
            f"Resolution: {width}x{height}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
            "Max Resolution Test"
        ]
        
        for i, text in enumerate(text_lines):
            cv2.putText(image, text, (50, 50 + i * 50), font, 1.5, (255, 255, 255), 3)
            cv2.putText(image, text, (50, 50 + i * 50), font, 1.5, (0, 0, 0), 2)
        
        # ファイル名を生成
        filename = f"max_res_camera{cam_id}_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # 画像を最高品質で保存
        success = cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        if success and os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"✅ カメラ{cam_id}: 保存成功")
            print(f"   📄 ファイル: {filename}")
            print(f"   📐 解像度: {width}x{height}")
            print(f"   📦 サイズ: {file_size / 1024:.1f} KB")
            captured_count += 1
        else:
            print(f"❌ カメラ{cam_id}: 保存失敗")
        
        # 少し待機
        time.sleep(0.1)
    
    print(f"\n📊 結果: {captured_count}/{len(test_resolutions)}枚の画像を保存しました")
    
    # 保存されたファイルを確認
    print("\n📋 保存されたファイル一覧:")
    try:
        files = os.listdir(output_dir)
        image_files = sorted([f for f in files if f.endswith('.jpg')])
        if image_files:
            total_size = 0
            for f in image_files:
                filepath = os.path.join(output_dir, f)
                file_size = os.path.getsize(filepath)
                total_size += file_size
                print(f"  📄 {f} ({file_size / 1024:.1f} KB)")
            print(f"  📦 総サイズ: {total_size / 1024:.1f} KB")
        else:
            print("  (画像ファイルなし)")
    except Exception as e:
        print(f"  エラー: {e}")
    
    # 権限も確認
    print(f"\n📂 ディレクトリ権限: {oct(os.stat(output_dir).st_mode)[-3:]}")
    
    return captured_count > 0

def test_directory_creation():
    """ディレクトリ作成テスト"""
    test_dirs = [
        '/tmp/captured_images',
        '/tmp/test_camera_output',
        os.path.expanduser('~/captured_images')
    ]
    
    print("\n📁 ディレクトリ作成テスト:")
    for test_dir in test_dirs:
        try:
            os.makedirs(test_dir, exist_ok=True)
            if os.path.exists(test_dir) and os.path.isdir(test_dir):
                print(f"✅ {test_dir}: 作成/アクセス成功")
            else:
                print(f"❌ {test_dir}: 作成失敗")
        except Exception as e:
            print(f"❌ {test_dir}: エラー - {e}")

if __name__ == "__main__":
    print("🚀 画像保存機能テストを開始します...")
    
    # ディレクトリ作成テスト
    test_directory_creation()
    
    # 画像保存テスト
    success = test_image_save_functionality()
    
    if success:
        print("\n✅ 画像保存機能テスト成功！")
        print("📁 保存場所: /tmp/captured_images")
    else:
        print("\n❌ 画像保存機能テスト失敗")
    
    print("\n🔍 ファイルを確認するには:")
    print("ls -la /tmp/captured_images/")
    print("file /tmp/captured_images/*.jpg")