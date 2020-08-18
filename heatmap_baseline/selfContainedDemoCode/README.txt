 
======================
About
======================
 
This demo is part of the paper: 

Visual Explanation by Interpretation: Improving Visual Feedback Capabilities of Deep Neural Networks.
Jose Oramas, Kaili Wang, and Tinne Tuytelaars.
International Conference on Learning Representations (ICLR) 2019
https://arxiv.org/abs/1712.06302

It contains models and associated data to experiments conducted on the ILSVRC'12, ILSVRC-Cats, MNIST and An8Flower datasets.
Please note that the released code is purely focused on the inference part of the method. For this reason, it might not be required to have GPU support when installing matconvet.

For more details, access to available models and the An8Flower dataset, please refer to the website of this project.
https://homes.esat.kuleuven.be/~joramas/projects/visualExplanationByInterpretation

======================
Dependencies
======================

This code depends on the following software packages
- Matconvet ( https://www.vlfeat.org/matconvnet/ )
- export_fig ( https://github.com/altmany/export_fig )

In addition, part of this code is based on the work "A Taxonomy and Library for Visualizing Learned Featuresin Convolutional Neural Networks" by Grun et al.
https://arxiv.org/abs/1606.07757

======================
Usage
======================

The contents of this file have the following structure

ROOT/
    - CNNModels
    - networkActivationsWeights
    - output
    - preStart.txt
    - README.txt
    - src
    - startup.m
    - testImages
    - visualizations
    
- For ease of use, it is recommended to start matlab from within the <ROOT> directory.
- Place in the directory <CNNModels> the matconvnet CNN models whose predictions will be explained.
  You may find pre-trained models associated to the paper in the project website.
- After installing matconvnet, the directory <src/matconvet/matconvnet_matlab> must link to the <matlab> directory inside your matconvnet installation.
- With the components above in place, you may execute the demo by running the following script from your matlab command line

    >> runDemo
  
- The directory <src> contains the source code of this demo.
- The directory <testImages> hosts images to be used as input to the models in the <CNNModels> directory.
  Feel free to add here any image you may want to test.
- The directory <networkActivationsWeights> hosts pre-computed matrices and associated data for the models to be explained.
  These matrices were the result of relevant feature selection as stated in Sec. 3.1 of the paper.
  Ideally, there is one of this matrices for every model in <CNNModels>.
- The directory <output> stores intermediate heatmaps (in -mat format) computed as part of the explanation procedure.
  These are the heatmaps that are later overlaid over the input image. They are useful in case a quantitative comparison or quality assessment is to be conducted.
- The directory <testImages> hosts images to be used as input to the models in the <CNNModels> directory.
  Feel free to add here any image you may want to test.
- In the directory <visualizations> the resulting visual explanations heatmaps will be stored in png format.
  Every visual explanation has two forms: i) a decoupled form; with separate visualizations for the top-3 relevant neurons with the strongest response in the given input image (e.g. Fig. 6 in the paper), and ii) a combined one, where the previous decoupled heatmaps are merged into one (e.g. Fig.8-Ours in the paper).

======================
Acknowledgements
======================

This work was partially suported by the FWO-SBO project SfS, the VLAIO RD-project SPOTT, the KU Leuven PDM Grant PDM/16/131, and a NVIDIA GPU grant. 

======================
Citing
======================

In case you use this code as part of your research/work we would appreciate if you could cite the associated paper:

@inproceedings{joramasAlICLR19,
  author = {Jos{\'e} {Oramas} and Kaili Wang and Tinne Tuytelaars},
  title = {Visual Explanation by Interpretation: Improving Visual Feedback Capabilities of Deep Neural Networks},
  booktitle = {ICLR},
  year = {2019}
}  
