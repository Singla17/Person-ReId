# Person Re-Identification
## Somanshu Singla [2018EE10314]

#### Environement Setup
For this project a conda based environment is recommeded, that can be created using the environment.yml file, because one module "faiss" has no official binary on pip.

```
conda env create -f environment.yml
```

This command prepares an environment named "reid_f" with python 3.7 and assumes only CPU [can't work with a GPU].     
For GPU based systems faiss-cpu can be uninstalled and faiss-gpu can be installed instead.   

#### File Structure
1. utils.py: contains various utilities used while implemeneting the project.
2. metrics.py: contains various metrics used for evaluation of models.
3. Demo.ipynb: Notebook for running the models in collab setup.
4. environment.yml: File to setup the environment.
5. model.py: contains the model implementations in Pytorch.
6. train.py: contains the code to train the models.
    ```
    python train.py --inp_path=<Path to trainset>  --out_path=<Path for model weights>  --num_classes=<Number of classes in trainset> --is_model_baseline=<"True" for baseline/"False" for improved>
    ```
7. run-test.py: contains the code to run the trained model on testset.
   ```
    python run-test.py --inp_path=<Path to testset>  --out_path=<Path for model weights>  --num_classes=<Number of classes in trainset> --is_model_baseline=<"True" for baseline/"False" for improved>
    ```
8. visualisation.py: contains code to get the visual form of model predictions i.e to get the images returned by model during retrieval.
    ```
    python visualisation.py --inp_path=<Path to testset>  --out_path=<Path for model weights>  --num_classes=<Number of classes in trainset> --is_model_baseline=<"True" for baseline/"False" for improved>  --vis_path=<Path to store the retrieved images>
    ```
9. visualisation_tSNE.py: contains the code to generate the feature point plot using PCA and tSNE.
    1. To use this script please install scikit-learn, seaborn, matplotlib
    ```
    python visualisation.py --inp_path=<Path to testset>  --out_path=<Path for model weights>  --num_classes=<Number of classes in trainset> --is_model_baseline=<"True" for baseline/"False" for improved>  --vis_path=<Path to store the retrieved images>
    ```


#### Models 
For baseline:
```
gdown https://drive.google.com/file/d/1FreqzErpm5WEriHhLzrhg8LESyxQY96N/view?usp=sharing
```
For improved model:
```
gdown https://drive.google.com/file/d/1mAIrVQaqjNqNMyEKggILjclXcwBRx_u2/view?usp=sharing
```

The Report can be found [here](./Report.pdf) and the Problem Statement can be found [here](./Problem_Statement.pdf).
