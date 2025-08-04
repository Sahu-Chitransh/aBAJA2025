from setuptools import find_packages, setup

package_name = 'lane_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_node = lane_detection.lane_node:main',
            'lane_node2 = lane_detection.lane_node2:main',
            'lane_node3 = lane_detection.lane_new:main',
            'lane_node4 = lane_detection.lane_new2:main',
            'lane_node5 = lane_detection.lane_new5:main',
            'lane_node6 = lane_detection.lane_new6:main',
            'steering_controller = lane_detection.steering_controller:main',
        ],
    },
)
