from setuptools import setup, find_packages

setup(
    name='image-toolbox',
    version='0.1.0',
    description='Helper functions for images',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Hashir Irfan, Ammar Aslam',
    author_email='hashirirfan15@gmail.com, ammaraslam10@gmail.com',
    url='https://github.com/sheikh-hashir/Image-Toolbox',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'imageio>=2.9.0',         # For reading/writing images
        'matplotlib>=3.3.0',      # For plotting
        'opencv-python>=4.5.0',   # For computer vision tasks
        'numpy>=1.19.0',          # For numerical operations
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
