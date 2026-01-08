#!/usr/bin/env python3
"""
ROS2バッグファイルをカラーからモノクロ（mono8）に変換するスクリプト
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from rosbags.rosbag2 import Reader, Writer
from rosbags.serde import deserialize_cdr, serialize_cdr
from sensor_msgs.msg import Image

def convert_to_mono(input_bag: str, output_bag: str):
    """
    ROS2バッグファイルをモノクロに変換
    
    Args:
        input_bag: 入力バッグファイルのパス
        output_bag: 出力バッグファイルのパス
    """
    print(f"Converting {input_bag} to mono8...")
    print(f"Output: {output_bag}")
    
    # 出力ディレクトリを作成
    Path(output_bag).parent.mkdir(parents=True, exist_ok=True)
    
    with Reader(input_bag) as reader, Writer(output_bag) as writer:
        # 接続情報をコピー
        conn_map = {}
        for conn in reader.connections:
            # 画像トピックのエンコーディングを mono8 に変更
            if "image" in conn.topic and conn.msgtype == "sensor_msgs/msg/Image":
                conn_map[conn.id] = writer.add_connection(
                    conn.topic.replace("image_raw", "image_mono"), 
                    conn.msgtype,
                    conn.serialization_format
                )
                print(f"  Mapping {conn.topic} -> {conn.topic.replace('image_raw', 'image_mono')}")
            else:
                conn_map[conn.id] = writer.add_connection(
                    conn.topic, 
                    conn.msgtype,
                    conn.serialization_format
                )
        
        converted_count = 0
        total_count = 0
        
        # メッセージを変換
        for conn, timestamp, rawdata in reader.messages():
            total_count += 1
            
            if "image" in conn.topic and conn.msgtype == "sensor_msgs/msg/Image":
                try:
                    # ROS2メッセージをデシリアライズ
                    msg = deserialize_cdr(rawdata, conn.msgtype)
                    
                    # BGRからモノクロに変換
                    if msg.encoding == 'bgr8':
                        # バイト配列からOpenCV画像に変換
                        img_array = np.frombuffer(msg.data, dtype=np.uint8)
                        img = img_array.reshape(msg.height, msg.width, 3)
                        
                        # グレースケールに変換
                        mono_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # メッセージを更新
                        msg.data = mono_img.tobytes()
                        msg.encoding = 'mono8'
                        msg.step = msg.width  # 1チャンネルなのでstep = width
                        
                        converted_count += 1
                        
                        # シリアライズして書き込み
                        new_data = serialize_cdr(msg, conn.msgtype)
                        writer.write(conn_map[conn.id], timestamp, new_data)
                        
                        if converted_count % 100 == 0:
                            print(f"  Converted {converted_count} images...")
                            
                    elif msg.encoding == 'mono8':
                        # すでにモノクロの場合はそのまま
                        writer.write(conn_map[conn.id], timestamp, rawdata)
                    else:
                        print(f"  Warning: Unsupported encoding {msg.encoding}, skipping...")
                        
                except Exception as e:
                    print(f"  Error converting image: {e}")
                    # エラーの場合は元データをそのまま書き込み
                    writer.write(conn_map[conn.id], timestamp, rawdata)
            else:
                # 画像以外はそのまま書き込み
                writer.write(conn_map[conn.id], timestamp, rawdata)
    
    print(f"Conversion complete!")
    print(f"  Total messages: {total_count}")
    print(f"  Converted images: {converted_count}")
    print(f"  Output: {output_bag}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 convert_bag_to_mono.py <input_bag> <output_bag>")
        print("Example: python3 convert_bag_to_mono.py ./my_slam_dataset ./my_slam_mono_dataset")
        sys.exit(1)
    
    input_bag = sys.argv[1]
    output_bag = sys.argv[2]
    
    # 入力ファイルの存在確認
    if not Path(input_bag).exists():
        print(f"Error: Input bag file does not exist: {input_bag}")
        sys.exit(1)
    
    try:
        convert_to_mono(input_bag, output_bag)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()