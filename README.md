# FrankaPandaPyBullet

## Requirements

Please finish reading this section before doing any installations!

To run the code you need to install PyBullet, to install PyBullet run the script

```[bash]
python3 -m pip install pybullet
```

Additionally, if you want to use [pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/index.html) run the following script

```[bash]
conda install -c conda-forge pinocchio
```

or you can follow the steps [here](https://stack-of-tasks.github.io/pinocchio/download.html).

Ideally, one should create a virtual environment and install everything inside. Since we are using `conda` to install `pinocchio`, we can create a virtual environment using conda

```[bash]
conda create --name ENV_NAME
```

for details of managing environments created using conda see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
