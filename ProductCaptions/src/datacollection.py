import os
from selenium import webdriver
from bs4 import BeautifulSoup
import urllib
import numpy as np
import pandas as pd
import os

"""
This module provided a few functions for webscraping data. It assumes
that you have your chromedrive installed and its path updated
It is a sample program for webscraping data and none of its dependencies
are provided in the requirements.txt file
"""
#Please modify this line to update the directory where you would like to place
#your webscraped data
os.pardir="/home/ubuntu/capturetocaption/data/raw/furniture1/"

def extracting_data_url(url, index_page):
    """
    Extracting html_source from the url and calls "extracting_data_from_html"
    to extract the images and the caption pairs and write it to file

    Please update the location of Chrome below or add it to your PATH
    """
    driver = webdriver.Chrome('/usr/local/bin/chromedriver') #update here
    #alternate encoding driver = webdriver.Chrome(executable_path=r'C:\Program Files\Chromedriver\chromedriver.exe')
    print(url)
    driver.get(url)
    html_source = driver.page_source
    extracting_data_from_html(html_source, index_page)

def extracting_data_from_html(html_source, index_page):
    """
    Extracting all the product images and caption pairs from the html source
    file of a particular amazon webpage and outputting the image caption
    pairs into a file
    """
    soup = BeautifulSoup(html_source, 'lxml')
    index_image=0 #index used in generating the filename of the image
    filelist=[] #list of filenames of the product images
    captions=[] #list of captions corresponding to each of the images provided in filelist
    #<span data-component-type="s-product-image" class="rush-component">
    for img in soup.find_all("span", {"data-component-type": "s-product-image"}):
                img_data=img.find('img')
                print(img_data['alt'])
                print(img_data['src'])
                print(img_data['data-image-index'])
                if(not(img_data['data-image-index']=="")):
                    filename="amazonfurniture_"+index_page+"_"+np.str(index_image)+".png"
                    urllib.request.urlretrieve(img_data['src'], os.pardir+filename)
                    captions.append(img_data['alt'])
                    filelist.append(filename)
                    index_image=index_image+1
    filedf=pd.DataFrame(list(zip(filelist, captions)), columns=["filename","caption"])
    # if file does not exist write header
    if not os.path.isfile(os.pardir+'FurnitureImageGeneration.csv'):
                filedf.to_csv(os.pardir+'FurnitureImageGeneration.csv')
    else: # else it exists so append without writing the header
                filedf.to_csv(os.pardir+'FurnitureImageGeneration.csv', mode='a', header=False)

def main():
            print("collecting data from amazon ...")
            numofindexpages=400
            for i in range(1,numofindexpages):
                driver = webdriver.Chrome() #driver = webdriver.Chrome(executable_path=r'C:\Program Files\Chromedriver\chromedriver.exe')
                url="https://www.amazon.com/s?k=sofa&i=garden&rh=n%3A3733551&page="+str(i)+"&qid=1567208242&ref=sr_pg_"+np.str(i)
                print(url)
                index_page = "page"+np.str(i)
                extracting_data_url(url, index_page)

if __name__ == "__main__":
            main()
