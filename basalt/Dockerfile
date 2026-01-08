FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

# 1. 依存ライブラリ
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential cmake git libgoogle-glog-dev libgflags-dev \
    libatlas-base-dev libsuitesparse-dev libcurl4-openssl-dev \
    libtbb-dev libglew-dev libopencv-dev libboost-all-dev \
    libeigen3-dev libepoxy-dev libpng-dev libjpeg-dev libtiff-dev \
    libfmt-dev libbz2-dev liblz4-dev libzstd-dev pkg-config x11-apps

# 2. Pangolin のビルド
WORKDIR /opt
RUN git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_CXX_FLAGS="-Wno-error=type-limits -w" && \
    make -j$(nproc) && make install

# 3. Basalt
COPY . /opt/basalt/
WORKDIR /opt/basalt/

# 4. 互換性パッチ（全階層のテスト読み込みを完全に無効化）
RUN find . -name "CMakeLists.txt" -exec sed -i 's/cmake_minimum_required(VERSION .*)/cmake_minimum_required(VERSION 3.22)/g' {} + && \
    find . -name "CMakeLists.txt" -exec sed -i 's/-Werror//g' {} + && \
    # 「test」という単語を含む add_subdirectory 命令をすべて無効化（デリミタを変えて安全に置換）
    find . -name "CMakeLists.txt" -exec sed -i 's|add_subdirectory(.*test.*)|#deleted|g' {} + && \
    # システム版Pangolinを使うよう調整
    sed -i 's|add_subdirectory(Pangolin)|#deleted|g' thirdparty/CMakeLists.txt && \
    # Eigenチェックの修正
    sed -i 's/find_package(Eigen3 3.4.0 EXACT REQUIRED)/find_package(Eigen3 REQUIRED)/g' CMakeLists.txt && \
    if [ -f cmake_modules/FindEigen3.cmake ]; then \
        sed -i 's/set(EIGEN3_VERSION_OK TRUE)/set(EIGEN3_INCLUDE_DIR \/usr\/include\/eigen3)\nset(EIGEN3_VERSION_OK TRUE)/g' cmake_modules/FindEigen3.cmake; \
    fi && \
    # Sophus パッチ
    find . -name "so2.hpp" -o -name "rxso2.hpp" | xargs -r sed -i 's/unit_complex().real()/unit_complex().x()/g' && \
    find . -name "so2.hpp" -o -name "rxso2.hpp" | xargs -r sed -i 's/unit_complex().imag()/unit_complex().y()/g'

# 5. ビルド実行
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_TESTS=OFF \
             -DEigen3_DIR=/usr/lib/cmake/eigen3 \
             -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
             -DCMAKE_CXX_FLAGS="-w" -DCMAKE_C_FLAGS="-w" && \
    make -j$(nproc) basalt_vio basalt_calibrate basalt_mapper basalt_opt_flow basalt_time_alignment basalt_calibrate_imu
