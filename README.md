# BoPro: Bootstrapping Webly Supervised Prototypical Learning with Visual-Semantic Alignment

## Abstract
Webly supervised learning (WSL) has attracted increasing attention for its impressive effectiveness and efficiency in exploring publicly accessible data at scale without manual annotation.
Most existing WSL methods,
however,
improve performance by scaling up the datasets with noisy web labels,
which provides suboptimal supervision.
Besides,
prior methods on noisy-label learning can only alleviate limited label errors and become subject to semantic misalignment noise,
where massive examples are with incorrect semantics and unknown concepts.
In this paper,
we propose BoPro, a prototypical learning method to effectively tackle various real-world noise on large-scale web data.
It exploits visual and semantic alignment in a unified supervised and contrastive learning framework.
First,
BoPro utilizes textual knowledge to pinpoint anchor prototypes whose visual contents are consistent with correct semantics,
which disambiguates for cluster regularization.
Second,
BoPro realizes collective bootstrapping in a manner of dictionary look-up to encourage smoother and wiser prediction reference by visually-similar instances.
Experiments on WebVision1k and NUS-WIDE demonstrate that BoPro handles realistic noise for learning representations under both single- and multi-label scenarios.
Moreover,
BoPro exhibits robustness to open-set recognition.
<!-- Codes and models will be available at \url{https://anonymous.4open.science/r/BoPro-F1D2}. -->

## Illustration of Cross-Modality Alignment and Collective Bootstrapping
![visualization](./imgs/1.png "We propose to explore cross-modality alignment (left) to reduce all kinds of noise including semantic noise, and collective bootstrapping (right) for label reference and regularization.")

## Overview of BoPro Architecture
![overview](./imgs/2.png "Overview of BoPro. Image encoders, projector, and classifiers are trained to learn the low-dimensional embedding space. Visual prototypes are initialized with anchors which are selected by matching denoised metadata to textual prototypes for semantic alignment. Prototypical contrastive learning is performed to constrain cluster distribution and visual prototypes are constantly polished with clean examples. In addition to instance-wise contrastive learning, the visual dictionary is exploited in collective bootstrapping where each of its key embeddings is matched to the query image for reference and regularization. Web labels are simultaneously adjusted to remove noise.")

## Dataset Download
In experiments, we use two large-scale web datasets with realistic noise: WebVision1k/Google500 and NUS-WIDE(Web).

### WebVision1k (WebVision 1.0)
The download link can be refered in <https://data.vision.ee.ethz.ch/cvl/webvision/download.html>.

~~We used the downsampled (256 * 256) version for convenience.~~

To improve performance, it is strongly encouraged to download the original sized version. Our preliminary experiments find that there exists a **noticeable** gap between the model trained on the resized and the original version.

Download the dataset into ```./dataset/webvision1k```.

### Google500 (subset of WebVision1k)
The Google500 dataset uses the randomly sampled 500 classes from the 1000 classes in WebVision1k with images only sourced from Google. The detailed description of Google500 can be refered in <https://github.com/bigvideoresearch/SCC>.

Google500 is only used for ablation study.

### ImageNet1k & ImageNet500
In experiments, we also report the performance on ImageNet datasets, which correspond to the same classes in WebVision1k and Google500. We evaluate webly-supervised models on the validation set.

ImageNet1k can be refered in <https://image-net.org/download.php>.

Download the dataset into ```./dataset/imagenet```.

### NUS-WIDE (Web)
The download link can be refered in <https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html>.
The official Web version of the NUS-WIDE is extracted from ```AllTags81.txt```. For each image, its user tags are compared with the chosen 81 concept tags to check if any of the 81 concept tag exists in the metadata of an image: 0 means ```non-exist```; 1 means ```exist```. These weak labels are not equivalent to the ground-truth labels which are provided in the folder ```AllLabels``` (e.g., ```Labels_airport.txt``` and ```Labels_animal.txt```).

We follow the official train/test split and these official web labels can be found respectively in ```./Train_Tags81.txt``` and ```Test_Tags81.txt```. We remove the images without any user tag from the 81 concepts. 

Download the dataset into ```./dataset/nus_wide_web```.

## Data Preparation

### TF-Record
In experiments, we use tfrecord format so that the I/O speed could be improved for training/evaluation.

Please check the ```./tfrecord/encode_tfrecord.py``` and fill in the root path of WebVision1k, ImageNet1k, and NUS-WIDE.

Please make sure the path is correct.

The dataset folder with examplar pathlist files can be downloaded in <https://drive.google.com/file/d/1r9s8OCsYQ4bkyG_66n9990LUxvFk5vcN/view?usp=share_link>. The results of tfrecord packaging should be similar to those exampler files. We refer to 


### WebVision1k/ImageNet1k & Google500/ImageNet1k filelist
The filelist can be referred in SCC <https://github.com/bigvideoresearch/SCC>.

For compatibility, we keep all image filelist in ```./dataset/webvision1k/filelist```.
* Text files that end with "_tf.txt" refer to the format in TF-Record.
* Text files that just end with ".txt" refer to the format in ".jpg" or ".jpeg".




<!-- ## Pretrained Weights
### BCNN (VGG16)
For experiments on fine-grained datasets, please use the ```--pretrained``` flag to load the pretrained weights of pytorch torchvision models.

### ResNet50
For experiments on large-scale datasets, please use the MoPro pretrained weights by downloading it from MoPro <https://github.com/salesforce/MoPro> and put the checkpoint weights as ```./ckpt_mopro/MoPro_V1_epoch90.tar```.

## Training
All the scripts can be found in ```./shells```.
### Few-Shot
Please replace the ```$pathlist_t``` with the corresponding path to the K-shot pathlist.
### Zero-Shot (Only Trained with Web Images)
Please remove the flag ```--use_fewshot``` in the script.

For example,
* use the script ```./shells/web-aircraft.sh``` for the training of BCNN models on web-aircraft.
* use the script ```./shells/webvision1k.sh``` for the training of ResNet models on WebVision1k.

## Evaluation

### Demo
All the scripts can be found in ```./eval_shells```.

For example,
* use the script ```./eval_shells/web-aircraft.sh``` for the evaluation of BCNN models on FGVC-Aircraft.
* use the script ```./eval_shells/webvision1k.sh``` for the evaluation of ResNet50 models on ImageNet1k.

### Model Weights
We provide the model weights in the ```./ckpt``` folder. Please check the evaluation shells for inference.

## Post-Processing
Enlightened by MoPro <https://openreview.net/forum?id=0-EYBhgw80y>, noise cleaning on the WebVision1k dataset can be performed to further reduce the noise and improve performance by fine-tuning.
For example,
* use the script ```./shells/webvision1k_ft.sh``` for noise cleaning and fine-tuning on WebVision1k with Mix-Up <https://arxiv.org/abs/1710.09412> strategy.

## Hyper-parameters Tuning
All the hyper-parameters are defined in ```./config_train.py```.

Preliminary experiments show that the $\beta=0.5=1-\alpha$ and $\gamma=0.6$ perform better than $\beta=0, 0.25, 0.75, 1$ and $\gamma=0.2$ on three fine-grained datasets (webFG496).

Other hyper-parameters are yet to be fine-tuned. Their current value is empirically set.

It remains to be explored which value of the distance threshold ```dist_th``` works best on picking out clean examples. One could design a threshold whose value varies with respect to epoch or loss.

## Results
The comparison with state-of-the-art methods on WebFG496 and WebVision1k/Google500 datasets demonstrates the effectiveness of FoPro in utilization of real-world fewshots.
![results](./imgs/img2.jpg "Results of comparison with SOTA.") -->


<!-- ## Citation
If you find this useful in your research, please consider citation of our work <https://arxiv.org/abs/2212.00465>:
```
@article{FoPro,
	title={FoPro: Few-Shot Guided Robust Webly-Supervised Prototypical Learning},
	author={Yulei Qin, Xingyu Chen, Chao Chen, Yunhang Shen, Bo Ren, Yun Gu, Jie
  Yang, Chunhua Shen},
	journal={AAAI},
	year={2023}
}
``` -->

