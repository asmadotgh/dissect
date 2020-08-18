## Installation
```bash
$ pip install -r requirements.txt
```

Setup python path to include repo
```
python setup.py develop
```

## Usage
1. Download the [**CelebA**](https://www.kaggle.com/jessicali9530/celeba-dataset) and [**3d-shapes**](https://github.com/deepmind/3d-shapes) dataset and create train and test fold for the classifier. 

If you already have them downloaded, it is easier to just run:
```
./python/prep_data.py --shapes --celeba --celeba_biased
```

2. Train a classifier. Skip this step if you have a pretrained classifier. The output of the classifier is saved at: $log_dir$/$name$. 

2.a. To train a binary classifier on shapes dataset
```
python train_classifier.py --config 'configs/redcyan_experiments/shapes_redcyan_Classifier.yaml'
```
2.b. To train a binary classifier on CelebA with 1 attribute, e.g. Smiling
```
python train_classifier.py --config 'configs/celeba64_biased_experiments/celebA_biased_Classifier.yaml'
```

3. Evaluate the binary classifier and export predicted probabilities. 
Process the output of the classifier and create input for Explanation model by discretizing the posterior probability.
The input data for the Explanation model is saved at: $log_dir$/$name$/explainer_input/
```
python test_classifier.py --config 'configs/redcyan_experiments/shapes_redcyan_Classifier.yaml' --n_bins=3

python test_classifier.py --config 'configs/celeba64_biased_experiments/celebA_biased_Classifier.yaml' --n_bins=10
```

4. Train discoverer model. The output is saved at: $log_dir$/$name$.

```
python train_discoverer.py --config 'configs/redcyan_experiments/shapes_redcyan_Discoverer.yaml'

python train_discoverer.py --config 'configs/celeba64_biased_experiments/celebA_biased_Discoverer_multidim.yaml'
```

for EPE models:

```
python train_explainer.py --config 'configs/redcyan_experiments/shapes_redcyan_Explainer.yaml'

python train_explainer.py --config 'configs/celeba64_biased_experiments/celebA_biased_Explainer.yaml'
```

for MEPE models:

```
python train_explainer.py --config 'configs/redcyan_experiments/shapes_redcyan_Explainer_multidim.yaml'

python train_explainer.py --config 'configs/celeba64_biased_experiments/celebA_biased_Explainer_multidim.yaml'
```

6. Save results of the trained Discoverer model for quantitative experiments and calculate evaluation metrics on it.
```
python evaluate_explainer_discoverer.py --config '[any explainer or discoverer config]'
```
