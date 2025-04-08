from setuptools import find_packages, setup

package_name = 'ashioto'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 
                      'gymnasium', 
                      'numpy', 
                      'stable-baseline3', 
                      'sb3-contrib', 
                      'pygame', 
                      'rl_zoo3',
                      'sbx-rl'],
    zip_safe=True,
    maintainer='usb',
    maintainer_email='binasyahdiba@gmail.com',
    description='A ROS2 package containing path planning using reinforcement learning',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ashioto = ashioto.ashioto_main:main'
        ],
    },
)
