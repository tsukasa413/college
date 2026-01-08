from setuptools import setup
import os
from glob import glob

package_name = 'quad_cam_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launchファイルをインストールするための設定
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Quad camera system for SLAM',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cam_node = quad_cam_system.cam_node:main',
            'sync_cam_node = quad_cam_system.sync_cam_node:main',
            'mono_cam_node = quad_cam_system.mono_cam_node:main',
            'max_resolution_capture = quad_cam_system.max_resolution_capture_node:main',
            'max_resolution_capture_service = quad_cam_system.max_resolution_capture_service:main',
        ],
    },
)
