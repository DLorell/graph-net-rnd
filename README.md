# graph-net-rnd
A repository for research and development of graph-neural-network based solutions for the grand Fleischer Imaging project.

The top level directories are individual projects. Enter any of them and then follow instructions below.


# Conda Env setup:

    $ conda create --name DESIREDENVNAME --file requirements/requirements.txt

# Pytorch/Torchvision Installation:

    $ conda activate DESIREDENVNAME
    $ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Running the program:

    $ conda activate DESIREDENVNAME
    $ python -m src

# For deveopers adding to the required python environment, update requirements.txt like so:

    $ conda activate ENVNAME
    $ conda list -e > requirements/requirements.txt