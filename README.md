# intro-ml-python
Playground project for learning purposes

# setup

## first time
```bash
conda create --name tf pip python=3.6 #create virtual environment
source activate tf  # enter created environment
pip install -r requirements.txt # inside environment install tensorflow and all dependencies

```
## next time

```bash
source activate tf  # enter created environment
# or 
conda activate tf
```

## occasionally 

```bash
pip install --upgrade tensorflow
pip freeze > requirements.txt 
pip install -r requirements.txt
pip install -r requirements.txt --upgrade
conda update conda
pip install --upgrade pip
pip install --upgrade jupyter
pip install --upgrade notebook
pip install --upgrade tensorboard

jupyter notebook #start jupyter

conda update -n base -c defaults conda
 


```
## tensor board
```bash
tensorboard --logdir func_approx/.tensorboard-func_approx.py
```

# Running from commandline

* Make sure `tf` environment is active
* Use `run.sh <path_to_python_script>`, for example

```bash
./run.sh func_approx/func_approx.py
```

* or add this to the root of the python script:
```python
import os,sys,inspect
currdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(currdir))
```

# modules and auto reloading
tl/df
```
%load_ext autoreload
%autoreload 2
# %aimport
```
https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html


## Setup IDE
Find the environment path:
```bash
conda info --envs | grep tf
```

### Intellij Ultimate Edition 
Open project settings and add python SDK pointing to proper virtual environment found above:
![Setup Intellij](docs/setupIdea.png)

And mark folder as sources.

### PyCharm Community Edition
![Setup Pycharm](docs/setupPycharm.png)


## Activation functions

![](docs/activations/sigmoid.png)

![](docs/activations/softsign.png)

![](docs/activations/tanh.png)

![](docs/activations/relu.png)

![](docs/activations/relu6.png)

![](docs/activations/leaky_relu.png)

![](docs/activations/crelu.png)

![](docs/activations/selu.png)

![](docs/activations/elu.png)

![](docs/activations/softplus.png)





