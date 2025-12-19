# カメラについて
## カメラ映像を見たいとき
```
sudo jetson_clocks
eCAM_argus_camera 
```
## カメラを増やしたいとき
```
cd ~/Documents/e-CAM80_CUOAGX/e-CAM80_CUOAGX/e-CAM80_CUOAGX_JETSON_AGX_ORIN_L4T36.3.0_04-JUNE-2024_R03/
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
cd ~/college/ros2_ws/
colcon build --symlink-install
source install/setup.bash
ros2 launch quad_cam_system multi_cam.launch.py
```
## bag
```
cd ~/college/ros2_ws/
rm -rf my_slam_dataset/
ros2 bag record -o my_slam_dataset \
  /camera_0/image_raw \
  /camera_1/image_raw \
  /camera_2/image_raw \
  /camera_3/image_raw \
  --storage mcap
```
```
cd ~/college/ros2_ws/
rosbags-convert --src ./my_slam_dataset/ --dst ./my_camera_dataset.bag
rm -rf ~/docker_sync/my_camera_dataset.bag
mv ./my_camera_dataset.bag ~/docker_sync/
```