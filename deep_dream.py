import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
from urllib2 import urlopen
import os
import zipfile
from helper import *
from optparse import OptionParser


def deep_dream(img_name, layers_inception):
    #download google's pre-trained neural network
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = 'data/'
    model_name = os.path.split(url)[-1]
    local_zip_file = os.path.join(data_dir, model_name)
    if not os.path.exists(local_zip_file):
        # Download
        model_url = urlopen(url)
        with open(local_zip_file, 'wb') as output:
            output.write(model_url.read())
        # Extract
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
  
    # start with a gray image with a little noise
    img_noise = np.random.uniform(size=(224,224,3)) + 100.0
  
    model_fn = 'tensorflow_inception_graph.pb'
    
    # Creating Tensorflow session and loading the model
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input':t_preprocessed})
    
    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
    
    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))
  
    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]
    resize = tffunc(np.float32, np.int32)(resize)

    def render_deepdream(t_obj, img0, layer, img_name, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

        # split the image into a number of octaves
        img = img0
        octaves = []
        for _ in range(octave_n-1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-resize(lo, hw)
            img = lo
            octaves.append(hi)
        
        # generate details octave by octave
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2])+hi
            for _ in range(iter_n):
                g = calc_grad_tiled(sess, img, t_grad, t_input)
                img += g*(step / (np.abs(g).mean()+1e-7))
            
            #this will usually be like 3 or 4 octaves
        #output deep dream image via matplotlib
        showarray(img/255.0, layer, img_name)

    channel = 139 # picking some feature channel to visualize
    
    img0 = PIL.Image.open(img_name)
    img0 = np.float32(img0)
     
    #Apply gradient ascent to that layer
    for layer in layers_inception:
        render_deepdream(tf.square(T(layer, graph)), img0, layer.split('_')[0][-2:], img_name.split('.')[0])
      
  
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-i', '--image', dest='image_loc', default='images/input/brad.jpg',
            help='path of imageto use for generating visualisation', metavar='IMAGE')
    parser.add_option('-l', '--layers', dest='layers', default='mixed3a_pool_reduce_pre_relu,mixed3b_pool_reduce_pre_relu,mixed4a_pool_reduce_pre_relu,mixed4b_pool_reduce_pre_relu,mixed5a_pool_reduce_pre_relu,mixed5b_pool_reduce_pre_relu',
            help='A comma seperated list of inception-v3 layers to use for deep dream', metavar='LAYERS')
    options, args = parser.parse_args()
    deep_dream(options.image_loc, options.layers.split(','))