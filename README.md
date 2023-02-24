# Gallery Search Based on OpenAI's Clip

# CLIP

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020) [[Model Card]](model-card.md) [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.



## Approach
![CLIP](https://user-images.githubusercontent.com/33936364/221297928-12591173-6557-419b-a108-d6a52f6e1ced.png)


## Installation

First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
$ pip install gradio==1.18
```

Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.

## Usage

To extract the features for the Upsplash dataset:
```
python extract_unsplash_features.py
```
Or you can download the Unsplash dataset ![features](https://github.com/sreenamadhu/clip-gallery-search/blob/main/unsplash/clip-ViT-B-32-unsplash-lite.pt) file directly. 

To run the gallery search demo, 
```
python gallery-search-demo.py
```

Here are some screenshots of sample output from gallery-search-demo.py:
1. For Image input:
<img width="1635" alt="Screen Shot 2023-02-24 at 3 13 21 PM" src="https://user-images.githubusercontent.com/33936364/221297327-c22a32e2-3883-4b6a-a461-73c39ac70108.png">
<img width="1657" alt="Screen Shot 2023-02-24 at 3 16 42 PM" src="https://user-images.githubusercontent.com/33936364/221297368-d911e80e-5176-4c67-97d7-49f8b96343f7.png">

2. For Textual input:
<img width="1618" alt="Screen Shot 2023-02-24 at 3 20 31 PM" src="https://user-images.githubusercontent.com/33936364/221297395-828619f4-a104-4055-95e8-4b22a573e57c.png">
<img width="1613" alt="Screen Shot 2023-02-24 at 3 23 26 PM" src="https://user-images.githubusercontent.com/33936364/221297399-9292a363-b78e-45f6-b1d8-80aa93599603.png">
