import os 
from glob import glob
from setuptools import setup

package_name = 'my_sphere_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'worlds'), glob(os.path.join('worlds', '*.world'))), 
        (os.path.join('share', package_name, 'rviz'),   glob(os.path.join('rviz', '*.rviz'))),  
        (os.path.join('share', package_name, 'models', 'sdf', 'sphere_goal'), glob(os.path.join('models','sdf','sphere_goal', '*.*'))),
        (os.path.join('share', package_name, 'models', 'urdf'), glob(os.path.join('models','urdf', '*.urdf'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='david',
    maintainer_email='david@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 'reader_mark_node = my_sphere_pkg.coordinate_node:main',
                             'my_client_node   = my_sphere_pkg.client_node:main',
        ],
    },
)
