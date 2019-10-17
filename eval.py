from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction().method4
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

#import matplotlib.pyplot as plt
#import cv2
import logging


class Evaluate:

    def __init__(self, file_to_log_results):
        smoothie = SmoothingFunction().method4
        self.add_score(file_to_log_results)
        logging.basicConfig(filename=file_to_log_results,level=logging.INFO)


    def add_score(self, file_to_log_results):
        #count=0
        total=[]
        total1=[]
        total2=[]
        total3=[]
        total4=[]
        for index in range(0, self.testdf.shape[0]):
            reference = [self.testdf['caption_new'][index].split()]
            candidate =self.testdf['gen_caption'][index].split()[1:-1]
            #print(len(candidate))
            if(len(candidate) >1):
                val1=sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
                val2=sentence_bleu(reference, candidate, weights=(0, 1, 0, 0), smoothing_function=smoothie)
                val3=sentence_bleu(reference, candidate, weights=(0, 0, 1, 0), smoothing_function=smoothie)
                val4=sentence_bleu(reference, candidate, weights=(0, 0, 0, 1), smoothing_function=smoothie)
                val=sentence_bleu(reference, candidate, smoothing_function=smoothie)
                total.append(val)
                total1.append(val1)
                total2.append(val2)
                total3.append(val3)
                total4.append(val4)
        logging.info(sum(total) / len(total) )
        logging.info(sum(total1) / len(total1) )
        logging.info(sum(total2) / len(total2) )
        logging.info(sum(total3) / len(total3) )
        logging.info(sum(total4) / len(total4) )
