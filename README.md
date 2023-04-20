# Pet BiFPN
Image segmentator for pet images from [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) using BiFPN (Bidirectonal Feature Pyramid Network) architecture with pretrained EfficientNet as backbone.  
Based on [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070).
## How to run
**Pytorch does not yet support Python 3.11 -- it's recommended to use Python 3.10.**  

Before using the script, make sure to install the dependencies:
```bash
pip install -r requirements.txt
```

To train a new model, simply run `python run.py --train`.  
To load a model from 'models/', run `python run.py --load <model filename>`.  
If you want to see result on your own image, put it in 'images/' and add flag `--input <image filename>` to the command.  
If you want to use GPU for training, add flag '--gpu'.  
