# rice-leaf-disease-classification

## overview

a rice leaf disease classification model made with EfficientNet-B0 backbone for my university project

## how to use?

- run `pip install -r requirements.txt` to install necessary libraries

- get the dataset from [this kaggle page](https://www.kaggle.com/competitions/paddy-disease-classification)

- move all folders inside `training_images` to a new folder named `data`

- run `python make_dataset.py` to create a structured dataset

- run `python train.py` to train the EfficientNet-B0 backbone. the trained model and metadata will be saved inside `checkpoints`

- run `python test.py` if you want to evaluate the model. the results will also be saved in `checkpoints`

## notes

- make sure the class order inside `prepare_data.py` matches with the model you are going to evaluate with `test.py`. view the model's `hyperparams.txt` for model details