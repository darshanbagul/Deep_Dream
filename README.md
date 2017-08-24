# Deep_Dream
Implementing deep dream using pretrained Google Inception-v3

## Dependencies

* numpy (http://www.numpy.org/)
* functools
* PIL (http://stackoverflow.com/questions/20060096/installing-pil-with-pip)
* tensorflow (https://www.tensorflow.org/install/#pip-installation)
* matplotlib (http://matplotlib.org/1.5.1/users/installing.html)
* urllib (https://pypi.python.org/pypi/urllib3)
* os
* zipfile

## Usage 

Once you have your dependencies installed via pip, run the demo script in terminal via

```
python deep_dream.py --image=<IMAGE_PATH> --layer=<LIST_OF_INCEPTION_LAYERS>
```

**Arguments:**

```
-i or --image 		
```
The path of input image

```
-l or --layers 		
```
List of all layers of inception-v3 network to test for generating deep dream images. 

eg.- mixed3a_pool_reduce_pre_relu, mixed4b_pool_reduce_pre_relu, etc.

Refer the file 'Inception_layers.txt' for names of layers in the Inception-v3 architecture


## Results

Experimenting across different activation layers/feature maps in the Inception-V3 architecture, I generated some interesting visualisations. I generated a GIF animation of the results for same image across different layers to view the projections of specific activatons of a given feature map on the input image. Let us visualise the results below:

**Brad Pitt**

![Image](https://github.com/darshanbagul/Deep_Dream/blob/master/images/results/brad.gif)

**Monalisa**

![Image](https://github.com/darshanbagul/Deep_Dream/blob/master/images/results/Monalisa.gif)

**Natural Scene**

![Image](https://github.com/darshanbagul/Deep_Dream/blob/master/images/results/scenery.gif)

**Sky**

![Image](https://github.com/darshanbagul/Deep_Dream/blob/master/images/results/sky.gif)


## Credits
	
1. Google Research blog on [Deep Dream](https://research.googleblog.com/2015/07/deepdream-code-example-for-visualizing.html)
2. Siraj Raval's excellent video on [Deep Dream in TensorFlow](https://www.youtube.com/watch?v=MrBzgvUNr4w)
