# iATW

## Setup (development)

The setup is platform-dependent. Currently supported are Linux and Windows. For GPU usage, you will need to have Cuda in version `11.x` installed.
First, recreate and activate the conda environment with the correct `environment-{platform}.yml`. For example, recreating the environment for Windows development:

```
conda env create --file environment-win.yml
conda activate iatw
```

Then, continue to install Pypi packages. Start with the `requirements.txt`, followed by `requirements-dev.txt`:

```
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Then, proceed with the correct `requirements-{platform}_{device}.txt` file to install `jax` and `jaxlib` for your platform (Linux or Windows) and device (CPU or GPU).
For example, installing for Windows and GPU using:

```
pip install -r requirements-win_gpu.txt
```
