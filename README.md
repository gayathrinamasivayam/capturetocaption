# Caption as a Service

## The aim of this project is to provide captions for products from their images. Online marketplaces are rapidly growing at a fast pace and the goal of this product is to provide instant image captioning system that can take an image of a furniture and produce an instant caption. The product is currently setup to work with sofas and chairs.

## The project currently is built on a modified encoder decoder architecture that has been used in the prior work on Image captioning. The project was built on Keras with a Tensorflow backend In the future I plan to use an attention architecture and vary it to see if I can improve the results obtained so far, as well as reduce the inference time of generating a caption from its image.  

### The code and the documentation are being updated on this repository

## Setup
Clone repository and update python path
```
repo_name=capturetocaption 
username=gayathrinamasivayam # Username for your personal github account
git clone https://github.com/$username/$repo_name
cd $repo_name
echo "export $repo_name=${PWD}" >> ~/.bash_profile
echo "export PYTHONPATH=$repo_name/src:${PYTHONPATH}" >> ~/.bash_profile
source ~/.bash_profile
```
