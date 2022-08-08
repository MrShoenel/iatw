# iATW

## Setup

First, recreate and activate the conda environment:

```
conda env create --file environment.yml
conda activate iatw
```

Then, continue to install Pypi packages. Start with the `requirements.txt`:

```
pip install -r requirements.txt
```

Then, proceed with the correct `requirements-{platform}_{device}.txt` file to install `jax` and `jaxlib` for your platform (Linux or Windows) and device (CPU or GPU).
For example, installing for Windows and GPU using:

```
pip install -r requirements-win_gpu.txt
```
