from setuptools import setup, find_packages

setup(
    name='integrated-adas',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='An application that integrates lane detection, road sign detection, drowsiness detection, and emotion recognition.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'opencv-python',
        'matplotlib',
        'pandas',
        'pygame',
        'moviepy',
        'tensorflow',  # or 'keras' if you're using Keras directly
        'scikit-learn',
        'scikit-image',
        'dlib',  # if using dlib for facial recognition
        'face_recognition'  # if using face recognition library
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)