# DISSECT
TensorFlow Implementation of [DISSECT: Disentangled Simultaneous Explanations via Concept Traversals](https://arxiv.org/abs/2105.15164).

This code is inspired by and built off of "Explanation by Progressive Exaggeration" ([code](https://github.com/batmanlab/Explanation_by_Progressive_Exaggeration), [paper](https://openreview.net/forum?id=H1xFWgrFPS)).

## Installation
```bash
$ pip install -r requirements.txt
```

Setup python path to include repo
```
python setup.py develop
```

## Usage
1. Download the [**CelebA**](https://www.kaggle.com/jessicali9530/celeba-dataset), [**3D Shapes**](https://github.com/deepmind/3d-shapes), and the newly generated [**SynthDerm**](https://mitmedialabaffectivecomputing.github.io/SynthDermDataset) dataset. Create train and test folders for the classifier.  

If you already have them downloaded, it is easier to just run:
```
./python/prep_data.py --shapes --celeba --celeba_biased --synthderm
```

2. Train a classifier. Skip this step if you have a pre-trained classifier. The output of the classifier is saved at: $log_dir$/$name$. 

2.a. To train a binary classifier on 3D Shapes dataset
```
python train_classifier.py --config 'configs/redcyan_experiments/shapes_redcyan_Classifier.yaml'
```
2.b. To train a binary classifier on SynthDerm dataset
```
python train_classifier.py --config 'configs/synthderm_experiments/synthderm_malignant_Classifier.yaml'
```

2.c. To train a binary classifier on CelebA with 1 attribute, e.g. Smiling
```
python train_classifier.py --config 'configs/celeba64_biased_experiments/celebA_biased_Classifier.yaml'
```

3. Evaluate the binary classifier and export predicted probabilities. 
Process the output of the classifier and create input for Explanation model by discretizing the posterior probability.
The input data for the Explanation model is saved at: $log_dir$/$name$/explainer_input/
```
python test_classifier.py --config 'configs/redcyan_experiments/shapes_redcyan_Classifier.yaml' --n_bins=3

python test_classifier.py --config 'configs/synthderm_experiments/synthderm_malignant_Classifier.yaml' --n_bins=2 --max_samples_per_bin=1350

python test_classifier.py --config 'configs/celeba64_biased_experiments/celebA_biased_Classifier.yaml' --n_bins=10
```

4. Train DISSECT discoverer model. The output is saved at: $log_dir$/$name$.

```
python train_discoverer.py --config 'configs/redcyan_experiments/shapes_redcyan_Discoverer_multidim.yaml'

python train_discoverer.py --config 'configs/synthderm_experiments/synthderm_malignant_Discoverer_multidim.yaml'

python train_discoverer.py --config 'configs/celeba64_biased_experiments/celebA_biased_Discoverer_multidim.yaml'
```

for EPE models:

```
python train_explainer.py --config 'configs/redcyan_experiments/shapes_redcyan_Explainer.yaml'

python train_explainer.py --config 'configs/synthderm_experiments/synthderm_malignant_Explainer.yaml'

python train_explainer.py --config 'configs/celeba64_biased_experiments/celebA_biased_Explainer.yaml'
```

for EPE-mod models:

```
python train_explainer.py --config 'configs/redcyan_experiments/shapes_redcyan_Explainer_multidim.yaml'

python train_explainer.py --config 'configs/synthderm_experiments/synthderm_malignant_Explainer_multidim.yaml'

python train_explainer.py --config 'configs/celeba64_biased_experiments/celebA_biased_Explainer_multidim.yaml'
```

for CSVAE models:

```
python train_csvae.py --config 'configs/csvae_experiments/shapes_redcyan_csvae_multidim.yaml'

python train_csvae.py --config 'configs/csvae_experiments/synthderm_malignant_csvae_multidim.yaml'

python train_csvae.py --config 'configs/csvae_experiments/celeba64_biased_or_csvae_multidim.yaml'
```

6. Save results of the trained Discoverer/Explainer/CSVAE model for quantitative experiments and calculate evaluation metrics on it.
```
python evaluate_models.py --config '[any explainer or discoverer config]'
```

## Reference
If you use this code or the released SynthDerm dataset, please reference [the following paper](https://arxiv.org/abs/2105.15164):
```
@article{ghandeharioun2021dissect,
  title={DISSECT: Disentangled Simultaneous Explanations via Concept Traversals},
  author={Ghandeharioun, Asma and Kim, Been and Li, Chun-Liang and Jou, Brendan and Eoff, Brian and Picard, Rosalind},
  journal={arXiv preprint arXiv:2105.15164},
  year={2021}
}
```

### Related Work
* Singla, S., Pollack, B., Chen, J., & Batmanghelich, K. (2020, September). [Explanation by Progressive Exaggeration](https://openreview.net/forum?id=H1xFWgrFPS). In International Conference on Learning Representations.
