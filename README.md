# Caption as a Service

### The aim of this project is to provide captions for products from their images. This is a novel idea for generating product captions from their images. Online marketplaces are rapidly growing at a fast pace and the goal of this product is to provide an instant image captioning system that can take an image of a product and produce an instant caption. We have currently trained the system using images of sofas and their captions from webscraped data.

![Contemporary bonded leather sofa with square base pillows](/download.png)
#### Generated Caption: "Contemporary bonded leather sofa with square base pillows"

### The captioning system uses an encoder decoder architecture. It is based on the prior work done in the field of image captioning [1,2] and has been modified for this particular use case. The project was built on Keras with a Tensorflow backend. It consists of a language LSTM encoder, an image encoder which consists of pre-trained VGG 16 fed into a dense layer. It uses an LSTM decoder. All parts of the encoder and decoder have been trained and learnt except for the features extracted from the last but one layer of VGG16 of the image. 

![Model](/model.png)

### The code and the documentation are being updated on this repository

## Setup
Create a new conda environment after downloading Anaconda, then clone the repository and install the depedencies
```
conda create -n captionrepo
conda activate captionrepo
git clone https://github.com/gayathrinamasivayam/capturetocaption.git
cd capturetocaption/
cd ProductCaption/
pip install --user -r requirements.txt
```
## Training (including data preprocessing)
To train and build the model on the sample dataset with predefined args and hyperparameters
```
cd src
python train.py
```
## Inference
To test an image of a sofa on the existing pre-trained model and tokenizer
```
cd src
python inference.py [--filepath <path_to_image_file> --filename <name_of_the_image_file>] 
```
#### References
#### [1] Tanti, M., Gatt, A., and Camilleri, K. P. (2018). Where to put the image in an image caption generator. Natural Language Engineering, 24(3):467–489.
#### [2] Ryan Kiros, Ruslan Salakhutdinov, Richard S. Zemel. "Multimodal Neural Language Models." ICML (2014).
#### [3] Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard S. Zemel, and Yoshua Bengio. 2015. Show, attend and tell: neural image caption generation with visual attention. In Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37 (ICML'15), Francis Bach and David Blei (Eds.), Vol. 37. JMLR.org 2048-2057.



