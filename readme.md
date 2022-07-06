# iATW

## Setup

First, recreate conda environment:

```
conda env create --file environment.yml
```

Then, continue to install Pypi packages (note: contains a pre-built wheel for `jaxlib` on Windows/Cuda 11.1; if you are on another platform, remove this line first and install `jaxlib` according to your platform):

```
pip install -r requirements.txt
```
