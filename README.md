# Caption as a Service

## The aim of this project is to provide captions for products from their images. Online marketplaces are rapidly growing at a fast pace and the goal of this product is to provide instant image captioning system that can take an image of a furniture and produce an instant caption. The product is currently setup to work with sofas and chairs.

## The project currently is built on a modified encoder decoder architecture that has been used in the prior work on Image captioning. The project was built on Keras with a Tensorflow backend In the future I plan to use an attention architecture and vary it to see if I can improve the results obtained so far, as well as reduce the inference time of generating a caption from its image.  

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
Create new development branch and switch onto it
```
branch_name=dev-readme_requisites-20180905 # Name of development branch, of the form 'dev-feature_name-date_of_creation'}}
git checkout -b $branch_name
```

## Initial Commit
Lets start with a blank slate: remove `.git` and re initialize the repo
```
cd $repo_name
rm -rf .git   
git init   
git status
```  
You'll see a list of file, these are files that git doesn't recognize. At this point, feel free to change the directory names to match your project. i.e. change the parent directory Insight_Project_Framework and the project directory Insight_Project_Framework:
Now commit these:
```
git add .
git commit -m "Initial commit"
git push origin $branch_name
```

## Requisites

- You will need the following packages installed: Python 3.6, Keras, NLTK, cv2

#### Dependencies

- [Streamlit](streamlit.io)

#### Installation
To install the package above, pleae run:
```shell
pip install -r requiremnts
```

## Build Environment
```
# Example

# Step 1
# Step 2
```

## Configs
- We recommond using either .yaml or .txt for your config files, not .json
- **DO NOT STORE CREDENTIALS IN THE CONFIG DIRECTORY!!**
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/)


## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

## Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
