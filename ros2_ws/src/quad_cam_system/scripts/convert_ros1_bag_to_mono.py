#!/usr/bin/env python3
"""
ROS1バッグファイルをカラーからモノクロ（mono8）に変換するスクリプト
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from rosbags.rosbag1 import Reader, Writer
import struct

def convert_ros1_to_mono(input_bag: str, output_bag: str):
    """
    ROS1バッグファイルをモノクロに変換
    
    Args:
        input_bag: 入力バッグファイルのパス (.bag)
        output_bag: 出力バッグファイルのパス (.bag)
    """
    print(f"Converting ROS1 bag {input_bag} to mono8...")
    print(f"Output: {output_bag}")
    
    # 出力ディレクトリを作成
    Path(output_bag).parent.mkdir(parents=True, exist_ok=True)
    
    with Reader(input_bag) as reader, Writer(output_bag) as writer:
        # 接続情報をコピー
        conn_map = {}
        for conn in reader.connections:
            # 画像トピックの場合はトピック名を変更
            if "image" in conn.topic and conn.msgtype == "sensor_msgs/Image":
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
        total_count = 0
        
        # メッセージを変換
        for conn, timestamp, rawdata in reader.messages():
            total_count += 1
            
            if "image" in conn.topic and conn.msgtype == "sensor_msgs/Image":
                try:
                    # ROS1 sensor_msgs/Image の手動パース
                    msg_dict = parse_sensor_msgs_image(rawdata)
                    
                    # BGRからモノクロに変換
                    if msg_dict['encoding'] == 'bgr8':
                        # バイト配列からOpenCV画像に変換
                        img_array = np.frombuffer(msg_dict['data'], dtype=np.uint8)
                        img = img_array.reshape(msg_dict['height'], msg_dict['width'], 3)
                        
                        # グレースケールに変換
                        mono_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # メッセージを更新
                        msg_dict['data'] = mono_img.tobytes()
                        msg_dict['encoding'] = 'mono8'
                        msg_dict['step'] = msg_dict['width']  # 1チャンネルなのでstep = width
                        
                        converted_count += 1
                        
                        # ROS1メッセージとして再構築
                        new_data = build_sensor_msgs_image(msg_dict)
                        writer.write(conn_map[conn.id], timestamp, new_data)
                        
                        if converted_count % 100 == 0:
                            print(f"  Converted {converted_count} images...")
                            
                    elif msg_dict['encoding'] == 'mono8':
                        # すでにモノクロの場合はそのまま
                        writer.write(conn_map[conn.id], timestamp, rawdata)
                    else:
                        print(f"  Warning: Unsupported encoding {msg_dict['encoding']}, skipping...")
                        
                except Exception as e:
                    print(f"  Error converting image: {e}")
                    # エラーの場合は元データをそのまま書き込み
                    writer.write(conn_map[conn.id], timestamp, rawdata)
            else:
                # 画像以外はそのまま書き込み
                writer.write(conn_map[conn.id], timestamp, rawdata)
    
    print(f"ROS1 conversion complete!")
    print(f"  Total messages: {total_count}")
    print(f"  Converted images: {converted_count}")
    print(f"  Output: {output_bag}")

def parse_sensor_msgs_image(data):
    """ROS1 sensor_msgs/Image メッセージをパース"""
    offset = 0
    
    # Header
    # seq (uint32)
    seq = struct.unpack('<I', data[offset:offset+4])[0]
    offset += 4
    
    # stamp.sec (uint32) 
    sec = struct.unpack('<I', data[offset:offset+4])[0]
    offset += 4
    
    # stamp.nsec (uint32)
    nsec = struct.unpack('<I', data[offset:offset+4])[0]
    offset += 4
    
    # frame_id length (uint32)
    frame_id_len = struct.unpack('<I', data[offset:offset+4])[0]
    offset += 4
    
    # frame_id (string)
    frame_id = data[offset:offset+frame_id_len].decode('utf-8')
    offset += frame_id_len
    
    # height (uint32)
    height = struct.unpack('<I', data[offset:offset+4])[0]
    offset += 4
    
    # width (uint32) 
    width = struct.unpack('<I', data[offset:offset+4])[0]
    offset += 4
    
    # encoding length (uint32)
    encoding_len = struct.unpack('<I', data[offset:offset+4])[0]
    offset += 4
    
    # encoding (string)
    encoding = data[offset:offset+encoding_len].decode('utf-8')
    offset += encoding_len
    
    # is_bigendian (uint8)
    is_bigendian = struct.unpack('<B', data[offset:offset+1])[0]
    offset += 1
    
    # step (uint32)
    step = struct.unpack('<I', data[offset:offset+4])[0]
    offset += 4
    
    # data length (uint32)
    data_len = struct.unpack('<I', data[offset:offset+4])[0]
    offset += 4
    
    # data (uint8[])
    image_data = data[offset:offset+data_len]
    
    return {
        'seq': seq,
        'sec': sec, 
        'nsec': nsec,
        'frame_id': frame_id,
        'height': height,
        'width': width,
        'encoding': encoding,
        'is_bigendian': is_bigendian,
        'step': step,
        'data': image_data
    }

def build_sensor_msgs_image(msg_dict):
    """ROS1 sensor_msgs/Image メッセージを構築"""
    data = b''
    
    # Header
    data += struct.pack('<I', msg_dict['seq'])
    data += struct.pack('<I', msg_dict['sec'])
    data += struct.pack('<I', msg_dict['nsec'])
    
    # frame_id
    frame_id_bytes = msg_dict['frame_id'].encode('utf-8')
    data += struct.pack('<I', len(frame_id_bytes))
    data += frame_id_bytes
    
    # Image fields
    data += struct.pack('<I', msg_dict['height'])
    data += struct.pack('<I', msg_dict['width'])
    
    # encoding
    encoding_bytes = msg_dict['encoding'].encode('utf-8')
    data += struct.pack('<I', len(encoding_bytes))
    data += encoding_bytes
    
    data += struct.pack('<B', msg_dict['is_bigendian'])
    data += struct.pack('<I', msg_dict['step'])
    
    # data
    data += struct.pack('<I', len(msg_dict['data']))
    data += msg_dict['data']
    
    return data

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 convert_ros1_bag_to_mono.py <input_bag> <output_bag>")
        print("Example: python3 convert_ros1_bag_to_mono.py my_slam_ros1.bag my_slam_mono.bag")
        sys.exit(1)
    
    input_bag = sys.argv[1]
    output_bag = sys.argv[2]
    
    # 入力ファイルの存在確認
    if not Path(input_bag).exists():
        print(f"Error: Input bag file does not exist: {input_bag}")
        sys.exit(1)
    
    try:
        convert_ros1_to_mono(input_bag, output_bag)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()