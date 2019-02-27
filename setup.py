from setuptools import setup

setup(name='image_utils',
      version='0.1',
      description='Simple but very useful image related utilities',
      url='https://github.com/sraashis/img_utils',
      author='Aashis Khanal',
      author_email='sraahis@gmail.com',
      license='MIT',
      packages=['img_utils'],
      install_requires=['numpy', 'PILLOW', 'scipy', 'opencv-python'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      zip_safe=False)
