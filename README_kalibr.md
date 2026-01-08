# how to create and run docker
```
FOLDER=~/docker_sync/
xhost +local:root
docker run -it --network=host \
    -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$FOLDER:/data" kalibr
```
# how to run docker
```
docker run -it --network=host \
    -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v ~/docker_sync:/data \
    kalibr_success:latest
```

# how to run kalibr

```
# 1. catkin_ws ディレクトリに移動
cd /catkin_ws

# 2. ROSの環境をセットアップ
source devel/setup.bash

# 3. (セットアップ成功後) Kalibrを実行
rosrun kalibr kalibr_calibrate_cameras \
    --bag /data/my_camera_dataset.bag \
    --target /data/april_6x6.yaml \
    --models ds-none ds-none ds-none ds-none \
    --topics /camera_0/image_raw /camera_1/image_raw /camera_2/image_raw /camera_3/image_raw \
    --approx-sync 0.005 \
    --bag-freq 4.0
```
