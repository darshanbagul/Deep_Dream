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

Arguments:
-i or --image 		The path of input image
-l or --layers 		List of all layers of inception-v3 network to test for generating
					deep dream images. eg.- mixed3a_pool_reduce_pre_relu, mixed4b_pool_reduce_pre_relu, etc.
					Refer the file 'Inception_layers.txt' for names of layers in the
					Inception-v3 architecture