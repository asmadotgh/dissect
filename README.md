# Explanation_by_Progressive_Exaggeration
Official Tensorflow implementation of ICLR 2020 paper: *Explanation By Progressive Exaggeration*.

[**Paper**](https://openreview.net/forum?id=H1xFWgrFPS)

<img src="./imgs/Model.png" height="50%" width="50%" >

## Experimental Results

### Qualitative Results
<img src="./imgs/Quality.jpg" height="50%" width="50%" >



## Installation
```bash
$ pip install -r requirements.txt
```

Setup python path to include repo
```
python setup.py develop
```

## Usage
1. Download the CelebA and [**3d-shapes**](https://github.com/deepmind/3d-shapes) dataset and create train and test fold for the classifier. 

If you already have them downloaded, it is easier to just run:
```
./python/prep_data.py --shapes --celeba --celeba_biased
```

2. Train a classifier. Skip this step if you have a pretrained classifier. The output of the classifier is saved at: $log_dir$/$name$. 

2.a. To train a binary classifier on shapes dataset
```
python train_classifier.py --config 'configs/shapes_redcolor_Classifier.yaml'
```
2.b. To train a binary classifier on CelebA with 1 attribute, e.g. Smiling
```
python train_classifier.py --config 'configs/celebA_Smile_Classifier.yaml'
```

3. Evaluate the binary classifier and export predicted probabilities. 
Process the output of the classifier and create input for Explanation model by discretizing the posterior probability.
The input data for the Explanation model is saved at: $log_dir$/$name$/explainer_input/
```
python test_classifier.py --config 'configs/shapes_redcolor_Classifier.yaml' --n_bins=3

python test_classifier.py --config 'configs/celebA_Smile_Classifier.yaml' --n_bins=10
```

4. Train discoverer model. The output is saved at: $log_dir$/$name$.

```
python train_discoverer.py --config 'configs/shapes_redcolor_Discoverer.yaml'

python train_discoverer.py --config 'configs/celebA_Smile_Discoverer.yaml'
```

for baseline models:

```
python train_explainer.py --config 'configs/shapes_redcolor_Explainer.yaml'

python train_explainer.py --config 'configs/celebA_Smile_Explainer.yaml'
```

for baseline++ models:

```
python train_explainer.py --config 'configs/shapes_redcolor_Explainer_multidim.yaml'

python train_explainer.py --config 'configs/celebA_Smile_Explainer_multidim.yaml'
```

6. Save results of the trained Discoverer model for quantitative experiments and calculate evaluation metrics on it.
```
python evaluate_explainer_discoverer.py --config 'configs/celebA_Smile_Explainer.yaml'
```


## Other (old repo - not using now)
If you want to follow where they get downloaded form and interactively see the data for CelebA: CelebA dataset is downloaded and saved at ./data/CelebA. IPython notebook creates text files with file names and labels and save them at ./data/CelebA/. These text files are used as input data to train the classifier.
```
./notebooks/Data_Processing.ipynb
```

To train a multi-label classifier on all 40 attributes
```
python train_classifier.py --config 'configs/celebA_DenseNet_Classifier.yaml'
```

To interactively process the output of the classifier and create input for Explanation model by discretizing the posterior probability.
The input data for the Explanation model is saved at: $log_dir$/$name$/explainer_input/
```
./notebooks/Process_Classifier_Output.ipynb
```

Use the saved results to perform experiments as shown in paper
```
./notebooks/Experiment_CelebA.ipynb 
```

# Cite

```
@inproceedings{
Singla2020Explanation,
title={Explanation  by Progressive  Exaggeration},
author={Sumedha Singla and Brian Pollack and Junxiang Chen and Kayhan Batmanghelich},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=H1xFWgrFPS}
}
```
