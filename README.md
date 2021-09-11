# machine-learning

Learn to create Machine Learning Algorithms in Python and R from two Data Science experts. Do note that this repository is tested on M1 macbook, hence the python package installation process is complicated.

## Installing conda

Firstly - I need to install Miniforge. The install script is on the GitHub page, or you can download it by clicking this link. It wanted to activate it in every terminal, which I didn't want so I turned that off by running:

```sh
conda config --set auto_activate_base false
```

Create a new conda environment with the command below and activate it afterwards.

```sh
conda env create --file=env.yaml --name tf_m1
conda activate tf_m1
```

Finally, I can install the packages all stated in requirements.txt. So far apyori and tensorflow is unavailable in conda, hence it is installed using pip command instead.

```sh
conda install -n tf_m1 --file reqconda.txt
pip install --no-dependencies -r reqpip.txt
```

## Python select interpretator VScode

1. Select Interpreter command from the Command Palette (Ctrl+Shift+P)

2. Search for "Select Interpreter"

3. Select the installed python directory (on tf_m1 for this case)
