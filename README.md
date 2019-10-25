# Caption as a Service

### The aim of this project is to provide captions for products from their images. This ia a novel idea for generating product captions from their images. Online marketplaces are rapidly growing at a fast pace and the goal of this product is to provide an instant image captioning system that can take an image of a product and produce an instant caption. We have currently trained the system using images of sofas and their captions from webscraped data.

![Contemporary bonded leather sofa with square base pillows](/download.png)
#### Generated Caption: "Contemporary bonded leather sofa with square base pillows"

### The captioning system uses an encoder decoder architecture. It is based on the prior work done in the field of image captioning. The project was built on Keras with a Tensorflow backend. It consists of a language LSTM encoder, an image encoder which consists of pre-trained VGG 16 fed into a dense layer. It uses an LSTM decoder.

![Model](/model.png)

### The code and the documentation are being updated on this repository

## Setup
Clone repository and update python path
```
git clone https://github.com/gayathrinamasivayam/capturetocaption.git

```
