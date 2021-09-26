# Crystalz

Scientific visualization of crystals with Streamlit + PyVista + Ipyvolume


## Requirements

Works with Python 3.x (tested on 3.8)

Install the requirements with `pip install -r requirements.txt`


## Getting data

You will need a directory with `.xyz` files; the `samples/` directory at the project root offers
a few simplistic examples.

For more fun, you can grab the dataset of the _Nomad2018_ Kaggle contest
(https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/data). Sign in, join the
competition and download the ZIP files. Then unpack all the `.xyz` file into a single directory,
and you are ready to go. Note that the all the files must be at the same level, the application
will not try to find them in subdirectories.


## Running the application

Because the Streamlit app file is not at the root of the project, running it involves a slightly
different command than usual:

```
$ python -m streamlit.cli run crystalz/viz/app.py -- --xyz-dir /your/xyz/directory
```

Usage is pretty straightforward, just play with the controls and enjoy. Computations are quite
slow, especially for big crystals; emphasis was on the visualizations and integration with
Streamlit.
