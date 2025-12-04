# カメラについて
## カメラ映像を見たいとき
```
sudo jetson_clocks
eCAM_argus_camera 
```
## カメラを増やしたいとき
```
sudo chmod +x ./install_binaries.sh
sudo -E ./install_binaries.sh
```
自動でRebootします
## カメラを確認する
リストで確認
```
sudo dmesg | grep -i "Detected eimx415 sensor"
```
どこに出力されているかで確認
```
ls /dev/video*
```

# camera multi launch
## service reset
```
# プロセスをクリーンアップ
sudo pkill -9 gst-launch-1.0
sudo pkill -9 python3

# カメラサービス再起動
sudo systemctl restart nvargus-daemon

# サービスの立ち上がりを少し待つ
sleep 3
```
## launch
```
sudo nvpmodel -m 0
sudo jetson_clocks
ls /dev/video*
```
```
colcon build --symlink-install
source install/setup.bash
ros2 launch quad_cam_system multi_cam.launch.py
```
## bag
```
ros2 bag record -o my_slam_dataset \
  /camera_0/image_raw \
  /camera_1/image_raw \
  /camera_2/image_raw \
  /camera_3/image_raw \
  --storage mcap
```