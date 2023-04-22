# Pet BiFPN
Image segmentator for pet images from [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) using BiFPN (Bidirectonal Feature Pyramid Network) architecture with pretrained EfficientNet as backbone.  
Based on [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070).

## Results
Model was trained using CUDA on entropy server provided by MIMUW.  
After 10 epochs on architecture with 2 BiFPN layers it achieved following results:

| Pet IOU | Background IOU | Outline IOU | Pet Accuracy | Background Accuracy | Outline Accuracy |
| --- | --- | --- | --- | --- | --- |
| 80.1% | 88.8% | 44.2% | 94.5% | 93.8% | 91.4% |

Here are some example results from the Oxford-IIIT Pet Dataset:
![example 1](https://raw.githubusercontent.com/Tsilkow/Pet_BiFPN/main/example_1.png)
![example 1](https://raw.githubusercontent.com/Tsilkow/Pet_BiFPN/main/example_2.png)
![example 1](https://raw.githubusercontent.com/Tsilkow/Pet_BiFPN/main/example_3.png)
![example 1](https://raw.githubusercontent.com/Tsilkow/Pet_BiFPN/main/example_4.png)

As we can see model has some difficulties with the details (paws are often classified as background), but the overall shape is classified correctly if the picture is not a close-up.

Here are some example results of pictures from outside the dataset (of a friendly dog Oscar):
![oskar 1](https://raw.githubusercontent.com/Tsilkow/Pet_BiFPN/main/oskar_1_result.png)
![oskar 2](https://raw.githubusercontent.com/Tsilkow/Pet_BiFPN/main/oskar_2_result.png)
![oskar 3](https://raw.githubusercontent.com/Tsilkow/Pet_BiFPN/main/oskar_3_result.png)
![oskar 4](https://raw.githubusercontent.com/Tsilkow/Pet_BiFPN/main/oskar_4_result.png)

Outside examples show even more better problems with classification, when only some of the dog is visible or if the picture is a close-up.

## How to run
**Pytorch does not yet support Python 3.11 -- it's recommended to use Python 3.10.**  

Before using the script, make sure to install the dependencies:
```bash
pip install -r requirements.txt
```

To train a new model, simply run `python run.py --train`.  
To load a model from 'models/', run `python run.py --load <model filename>`.  
If you want to see result on your own image, put it in 'images/' and add flag `--input <image filename>` to the command.  
If you want to use GPU for training, add flag `--gpu`.  
