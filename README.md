# nbee

A pytorch based framework for medical image processing with COnvolutional Neural Network. 
Along with example with unet for DRIVE dataset segmentation [1]. DRIVE dataset is composed of 40 retinal fundus images. 

### Required dependencies

We need python3, numpy, pytorch, torchvision, matplotlib and PILLOW packages

```
pip install -r ature/assets/requirements.txt
```



### Project Structure

* [ature/nbee](https://github.com/sraashis/ature/tree/master/nbee) nbee framework core.
* [ature/utils](https://github.com/sraashis/ature/tree/master/utils) Utilities for dealing with F1-score, image cropping, slicing, visual precision-recall, auto split train-validation-test set and many more.
* [ature/viz](https://github.com/sraashis/ature/tree/master/viz) Easy pytorch visualization.
* [ature/testarch](https://github.com/sraashis/ature/tree/master/nbee) Full end to end working [u-net(Olaf Ronneberger et al.)](https://arxiv.org/abs/1505.04597) 
for retinal image segmentation.

## Dataset check
![alt text](https://github.com/sraashis/ature/blob/master/data/DRIVE/images/01_test.tif)

## References

1. J. Staal, M. Abramoff, M. Niemeijer, M. Viergever, and B. van Ginneken, “Ridge based vessel segmentation in color
images of the retina,” IEEE Transactions on Medical Imaging 23, 501–509 (2004)
