import os 
from glob import glob
from setuptools import setup

package_name = 'my_environment_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'),  glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'rviz'),    glob(os.path.join('rviz', '*.rviz'))),    
        (os.path.join('share', package_name, 'worlds'),  glob(os.path.join('worlds', '*.world'))),    



    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='david',
    maintainer_email='david@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                           'run_environment = my_environment_pkg.run_environment:main',
        ],
    },  
)
