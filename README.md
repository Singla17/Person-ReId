# Person Re-Identification
## Somanshu Singla [2018EE10314]

#### Environement Setup
For this project a conda based environment is recommeded, that can be created using the environment.yml file, because one module "faiss" has no official unooficial binary on pip.

```
conda env create -f environment.yml
```

This command prepares an evivornment named "reid_f" with python 3.7 and assumes only CPU [can't work with a GPU].     
For GPU based systems faiss-cpu can be uninstalled and faiss-gpu can be installed instead.   