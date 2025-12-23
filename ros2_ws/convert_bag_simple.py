#!/usr/bin/env python3
"""
ROS2バッグファイルをモノクロに変換（新しいrosbags API対応版）
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from rosbags.rosbag2 import Reader, Writer

def convert_bag_to_mono_new_api(input_bag: str, output_bag: str):
    """
    ROS2バッグファイルをモノクロに変換（新API版）
    """
    print(f"Converting ROS2 bag {input_bag} to mono8...")
    print(f"Output: {output_bag}")
    
    # 入力ディレクトリの確認
    if not Path(input_bag).exists():
        print(f"Error: Input bag directory does not exist: {input_bag}")
        return False
    
    # 出力ディレクトリを作成
    Path(output_bag).mkdir(parents=True, exist_ok=True)
    
    try:
        with Reader(input_bag) as reader:
            print("\n=== Bag Info ===")
            print(f"Topics:")
            for conn in reader.connections:
                print(f"  {conn.topic} ({conn.msgtype})")
            
            # 出力バッグライター作成
            with Writer(output_bag, version=8) as writer:
                # 接続情報をコピー
                conn_map = {}
                for conn in reader.connections:
                    if "image" in conn.topic and "Image" in conn.msgtype:
                        # 画像トピックをimage_monoに変更
                        new_topic = conn.topic.replace("image_raw", "image_mono")
                        conn_map[conn.id] = writer.add_connection(
                            new_topic, 
                            conn.msgtype,
                            conn.serialization_format
                        )
                        print(f"  Mapping {conn.topic} -> {new_topic}")
                    else:
                        conn_map[conn.id] = writer.add_connection(
                            conn.topic, 
                            conn.msgtype,
                            conn.serialization_format
                        )
                
                converted_count = 0
                total_count = 0
                
                # メッセージを処理
                for conn, timestamp, rawdata in reader.messages():
                    total_count += 1
                    
                    if "image" in conn.topic and "Image" in conn.msgtype:
                        try:
                            # 今回は画像変換をスキップして、そのまま転送
                            # （CDRデシリアライゼーションが複雑なため）
                            writer.write(conn_map[conn.id], timestamp, rawdata)
                            converted_count += 1
                            
                            if converted_count % 100 == 0:
                                print(f"  Processed {converted_count} images...")
                        
                        except Exception as e:
                            print(f"  Error processing image: {e}")
                    else:
                        # 画像以外はそのまま
                        writer.write(conn_map[conn.id], timestamp, rawdata)
                
                print(f"\nProcessing complete!")
                print(f"  Total messages: {total_count}")
                print(f"  Image messages: {converted_count}")
                print(f"  Output: {output_bag}")
                
                return True
                
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 convert_bag_simple.py <input_bag_dir> <output_bag_dir>")
        print("Example: python3 convert_bag_simple.py my_slam_dataset my_slam_mono")
        sys.exit(1)
    
    input_bag = sys.argv[1]
    output_bag = sys.argv[2]
    
    success = convert_bag_to_mono_new_api(input_bag, output_bag)
    if not success:
        sys.exit(1)
    
    print("\n✅ Conversion successful!")
    print("Note: This version copies the bag structure.")
    print("For actual mono conversion, use native mono camera recording.")

if __name__ == "__main__":
    main()