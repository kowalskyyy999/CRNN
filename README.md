# CRNN
Re-Implementation [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717) using Pytorch Framework. The dataset [Synthetic dataset](https://www.robots.ox.ac.uk/~vgg/data/text/#sec-synth)

## Download dataset
```bash
cd data
bash download_synth90k.sh
```

## Training
```python
python train.py
```

## Statistic

1. **Pie Chart**
![pie-chart](https://user-images.githubusercontent.com/51861035/192148174-a988ddd2-fb26-4c9b-b65b-a29cb4fb397e.jpg)

2. **Histogram**
![histogram](https://user-images.githubusercontent.com/51861035/192148368-32a2c593-5f1a-447f-b939-c14d124ca501.jpg)

## Prediction
 ![prediction 1](https://user-images.githubusercontent.com/51861035/192148413-cebaf164-dc13-4535-9c9f-59bcea4b4df0.jpg)


![prediction 2](https://user-images.githubusercontent.com/51861035/192148534-d097ae69-b82b-4f47-b0e0-73f76b07017e.jpg)


## REFERENCE
* https://github.com/GitYCC/crnn-pytorch
* https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/