
# Glottal excitation model with Keras and Theano

This code provides basic tools for predicting glottal excitation waveforms (or spectra) from acoustic features.

## Dependencies:
- theano
- keras
- numpy
- sklearn (data normalization)
- scipy (reading NetCDF files)
- matplotlib

## Data 

Currently, data is only provided in pre-packaged format. 
Some data (approx. 650MB) can be found here
https://drive.google.com/open?id=0B5M0_b2kZdj6SHNUSnEzN1hwZzg

The zip-archive contains 10 `data.nc` files, each containing acoustic features and corresponding excitation waveforms from 100 speech utterances.

The example data is derived from the Hurricane Challenge corpus.
If you intend to use it, please see the licence conditions at 
http://www.cstr.ed.ac.uk/projects/hurricane/1/index.html

## Keras configuration

The code uses Theano backend and assumes "channels last" tensor dimension ordering. 

Check your Keras config at 
`~/.keras/keras.json`
and ensure it has

```
{
    "image_dim_ordering": "th", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "theano"
}
```

## Training the model

Assuming the `data.nc` files are located at `./traindata/`, you can run an example training with

```bash
python main.py --mode train --data ./traindata/ --rnn_context_len=1 --max_files=2
```

Note that the first data file is used for validation and the rest for training. Training can be run with more data and longer context length for RNN input with, for example

```bash
python main.py --mode train --data ./traindata/ --rnn_context_len=128
```
## Generating 

For the toy example above, you can generate spectra to `./output/` by running

```bash
python main.py --mode generate --max_files=1 --rnn_context_len=1 --output=./output/ --testdata=./traindata/
```

When using different `--rnn_context_len`, modify the call accordingly.

## Data format

Data is packaged in NetCDF files. For inspecting data in, say, `data.nc1` you can

```python
from scipy.io import netcdf
fid = netcdf.netcdf_file('data.nc1','r')

outputs = fid.variables['targetPatterns'][:].copy()
# read output mean and standard deviation
m = fid.variables['outputMeans'][:].copy()
s = fid.variables['outputStdevs'][:].copy()
# de-normalize data
outputs = s * outputs + m

# read input acoustic features
inputs = fid.variables['inputs'][:].copy()
m = fid.variables['inputMeans'][:].copy()
s = fid.variables['inputStdevs'][:].copy()
inputs = s * inputs + m
```

See details in `data_utils.py`

## Related work

We used the same basic toolset for Generative Adversarial Net -based excitation pulse generation. 
Check out Bajibabu's repo at https://github.com/bajibabu/GlottGAN

## Known issues

With the tools provided here, it's difficult to create your own data or actually synthesize anything. For this, GlottDNN vocoder and related tools are required (and will be published soon). If you come up with a nice model and want to hear how it sounds, please contact the author by email.

## Licence

This project is licensed under the terms of the MIT license.
Copyright 2017 Lauri Juvela (lauri.juvela@aalto.fi)

See LICENCE.md for full licence