from setuptools import setup

setup(name='nnbee',
      version='0.1',
      description='Elegant Convolutional Neural Network (CNN) bsed on pytorch Framework',
      url='https://github.com/sraashis/ature',
      download_url='https://github.com/sraashis/ature/releases/tag/STABLE',
      author='Aashis Khanal',
      author_email='sraahis@gmail.com',
      license='MIT',
      packages=['nnbee'],
      install_requires=['numpy', 'PILLOW', 'scipy', 'opencv-python', 'torch', 'torchvision'],
      classifiers=[
          "Programming Language :: Python :: 3",
          'License :: OSI Approved :: MIT License',
          "Operating System :: OS Independent",
      ],
      zip_safe=True)
