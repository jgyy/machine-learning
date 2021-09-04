# machine-learning

Learn to create Machine Learning Algorithms in Python and R from two Data Science experts

## Installing conda

Firstly - I need to install Miniforge. The install script is on the GitHub page, or you can download it by clicking this link. It wanted to activate it in every terminal, which I didn't want so I turned that off by running:

```sh
conda config --set auto_activate_base false
```

This is pretty much the same as creating a virtual environment with Python, just using a different tool. Like with Python, the virtual environment then needs to be activated:

```sh
conda activate
```

Finally, I can install Scikit-Learn:

```sh
conda install --file requirements.txt
```

## Python select interpretator

1. Select Interpreter command from the Command Palette (Ctrl+Shift+P)

2. Search for "Select Interpreter"

3. Select the installed python directory
