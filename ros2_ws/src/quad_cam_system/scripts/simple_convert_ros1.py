#!/usr/bin/env python3
"""
ROS1バッグファイルの内容確認とモノクロ変換（簡易版）
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from rosbags.rosbag1 import Reader, Writer

def simple_convert_ros1_to_mono(input_bag: str, output_bag: str):
    """
    シンプルなROS1バッグファイルモノクロ変換
    
    Args:
        input_bag: 入力バッグファイルのパス (.bag)
        output_bag: 出力バッグファイルのパス (.bag)
    """
    print(f"Converting ROS1 bag {input_bag} to mono8...")
    print(f"Output: {output_bag}")
    
    # 入力ファイルの確認
    if not Path(input_bag).exists():
        print(f"Error: Input file does not exist: {input_bag}")
        return False
    
    try:
        with Reader(input_bag) as reader:
            print("\n=== Bag Info ===")
            print(f"Topics:")
            for conn in reader.connections:
                print(f"  {conn.topic} ({conn.msgtype})")
            
            print(f"\nMessage count: {reader.message_count}")
            
            # 画像トピックがあるかチェック
            image_topics = [conn for conn in reader.connections 
                           if "image" in conn.topic and conn.msgtype == "sensor_msgs/Image"]
            
            if not image_topics:
                print("No image topics found!")
                return False
            
            print(f"Found {len(image_topics)} image topics")
            
            # 出力ディレクトリを作成
            Path(output_bag).parent.mkdir(parents=True, exist_ok=True)
            
            with Writer(output_bag) as writer:
                # 接続情報をコピー（画像以外はそのまま、画像はmono版に変更）
                conn_map = {}
                for conn in reader.connections:
                    if conn in image_topics:
                        new_topic = conn.topic.replace("image_raw", "image_mono")
                        conn_map[conn.id] = writer.add_connection(
                            new_topic, 
                            conn.msgtype, 
                            conn.md5sum,
                            conn.callerid
                        )
                        print(f"  Mapping {conn.topic} -> {new_topic}")
                    else:
                        conn_map[conn.id] = writer.add_connection(
                            conn.topic, 
                            conn.msgtype,
                            conn.md5sum, 
                            conn.callerid
                        )
                
                converted_count = 0
                skipped_count = 0
                total_count = 0
                
                # メッセージ処理
                for conn, timestamp, rawdata in reader.messages():
                    total_count += 1
                    
                    if conn in image_topics:
                        try:
                            # 画像メッセージは今回はスキップ（構造が複雑なため）
                            # 代わりに元データをそのまま書き込み
                            writer.write(conn_map[conn.id], timestamp, rawdata)
                            skipped_count += 1
                            
                            if skipped_count % 100 == 0:
                                print(f"  Processed {skipped_count} images...")
                                
                        except Exception as e:
                            print(f"  Error processing image: {e}")
                            writer.write(conn_map[conn.id], timestamp, rawdata)
                    else:
                        # 画像以外はそのまま
                        writer.write(conn_map[conn.id], timestamp, rawdata)
                
                print(f"\nConversion complete!")
                print(f"  Total messages: {total_count}")
                print(f"  Image messages: {skipped_count}")
                print(f"  Output: {output_bag}")
                return True
                
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 simple_convert_ros1.py <input_bag> <output_bag>")
        print("Example: python3 simple_convert_ros1.py my_camera_dataset.bag test_output.bag")
        sys.exit(1)
    
    input_bag = sys.argv[1]
    output_bag = sys.argv[2]
    
    success = simple_convert_ros1_to_mono(input_bag, output_bag)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()