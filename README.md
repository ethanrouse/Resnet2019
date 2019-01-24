# Resnet2019
Resnet implementation using code from https://github.com/raghakot/keras-resnet as a base. All neccessary project files will be stored here for future research.

##Before doing any programming 
  Install [Anaconda](https://www.anaconda.com/download/)
  Install necessary [GPU packages/tools](https://www.tensorflow.org/install/gpu)*
    Make sure to install CUDA 9.0

##Creating and running a Python environment* 
  Using the [environment file](https://github.com/ethanrouse/Resnet2019/blob/master/resnetenv.yml) 
    '''conda env create --file resnetenv.yml 
    python -m ipykernel install --user --name resnetenv --display-name "Python (resnetenv)"'''
  Manual: 
    '''conda create -n kerasenv python=3.6.5 
    activate kerasenv
    conda install nb_conda_kernels 
    pip install tensorflow_gpu 
    pip install keras 
    python -m ipykernel install --user --name resnetenv --display-name "Python (resnetenv)"''' 
  For activating/deactivating the environment use the following commands 
    '''activate resnetenv'''
    '''deactivate''' 
  While the environment is activated, Jupyter Notebooks can then be run with 
    '''jupyter notebook'''

###[Tutorial on Jupyter Notebooks](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)


*The above steps can be altered to use CPU rather than GPU computation but in most cases the GPU will be necessary with the volume of data that comes with MRIs. 
    To use a cpu based tensorflow simply do the following commands while in the activated environment 
    '''pip uninstall tensorflow-gpu 
    pip install tensorflow'''
