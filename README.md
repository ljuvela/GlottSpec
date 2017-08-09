This project is licensed under the terms of the MIT license.
Copyright 2017 Lauri Juvela (lauri.juvela@aalto.fi)

Glottal excitation model with Keras and Theano

Dependencies:
	- theano
	- keras
	- numpy
	- sklearn (data normalization)
	- scipy (reading NetCDF files)
	- matplotlib

This code uses Theano backend and assumes "channels last" tensor dimension ordering. 

Check your Keras config at 
~/.keras/keras.json

and ensure it has

{
    "image_dim_ordering": "th", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "theano"
}
