import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import shutil
import re

class DataPreprocessing:

    def __init__(self):
        #some of the popular brands on amazon selling sofas
        brands= ["24 7 Shop at Home",
        "Acme Furniture",
        "America Luxury Sofa",
        "American Eagle Furniture",
        "Apt2B",
        "Armen Living",
        "Baxton Studio",
        "Benjara",
        "Benzara",
        "Blackjack Furniture",
        "Boca Rattan",
        "BOWERY HILL",
        "Brika Home",
        "Chelsea Home",
        "Classic Brands"
        "Christopher Knight Home",
        "Coaster Home Furnishings",
        "Container Furniture Direct",
        "DIVANO ROMA FURNITURE",
        "DHP Emily",
        "Dreamseat",
        "Epic Furnishings",
        "Ethan Allen",
        "Flash Furniture",
        "FDW",
        "Furniture of America",
        "Global Furniture USA",
        "Glory Furniture",
        "Great Deal Furniture",
        "Homelegance",
        "HOMES: Inside Out",
        "Iconic Home",
        "Istikbal",
        "J and M Furniture",
        "Jennifer Taylor Home",
        "Kardiel",
        "Lesro",
        "Limari Home",
        "Meridian Furniture",
        "Modway",
        "Offex",
        "DG Casa",
        "Oadeer Home",
        "Pemberly Row",
        "Poundex",
        "Rivet",
        "Rivet Revolve"
        "Serta",
        "Signature Design by Ashley",
        "Simmons Upholstery",
        "SOUTH CONE HOME",
        "Stone and Beam",
        "Sunset Trading",
        "TK Classics",
        "TrueModern",
        "US Pride Furniture",
        "VVR Homes",
        "Zinus Ricardo",
        "Zinus Jackie",
        "Flash Furniture Benchcraft Maier",
        "Ashley Furniture Signature Design",
        "Serta RTA Palisades Collection",
        "Homelegance Resonance",
        "Best Choice Products",
        "Serta Rane Collection",
        "Zinus Lauren",
        "Madison Home",
        "TLY Cotton Karlstad",
        "AODAILIH",
        "Yaheetech"
        ]
        brands=[w.lower()  for w in brands ]

        os.pardir ="C:\\Users\\Gayathri\\Documents\\Insight\\ImageCaption\\capturetocaption\\data\\raw\\sofas\\"
        self.inputdf = pd.DataFrame() #dataframe with raw data
        self.outputdf=pd.DataFrame(columns=['filename','caption']) #dataframe with preprocessed captions and filename pairs

        self.len_traindir = 8000 #currently set to 8000

        self.load_file()
        #self.makedir()

    #load the raw data into the inputdf dataframe
    def load_file(self):
        os.filename= os.pardir+"FurnitureImageGeneration.csv"
        self.inputdf = pd.read_csv(os.filename)
        self.inputdf=self.inputdf.drop(columns=["Unnamed: 0"])

    #dispaly the an image and its caption using an index from the inputdf
    def display_image_caption(self, index):
        imagefile=self.inputdf['filename'][index]
        print(imagefile)
        img = cv2.imread(os.pardir+imagefile)
        plt.imshow(img)
        print(self.inputdf['caption'][index])

    def __input_dataframe_size(self):
        return (self.inputdf.shape[0])

    #Create the train and test directories
    #can be improved to shuffle and store random data
    def maketrain_test_dir(self):
        train_dataset_dir= os.pardir+"train1\\"
        test_dataset_dir= os.pardir+"test1\\"
        os.mkdir(train_dataset_dir)
        os.mkdir(test_dataset_dir)
        print(self.inputdf.shape[0])
        #len_trainset= abs(self.__input_dataframe_size()*0.8)
        for i, filename in enumerate(self.inputdf[0:self.len_traindir]['filename']):
            print(filename)
            src=os.pardir+filename
            dst=train_dataset_dir+filename
            shutil.copyfile(src, dst)
        for i, filename in enumerate(df[self.len_traindir:]['filename']):
            print(filename)
            src=os.pardir+filename
            dst=test_dataset_dir+filename
            shutil.copyfile(src, dst)
        #df_train_captions=df['caption'][0:len_trainset]
        #df_test_captions=df['caption'][len_trainset:]

    # internal funtion to take a word and transform it
    def __preprocess_word(self, word):
        num2words = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten'}
        if word.isdigit():
            if int(word) in num2words.keys():
                return (num2words[int(word)])
            else:
                return ""
        else:
            if word.isalpha():
                return word
            else:
                if '-' in word:
                    #split the words by - and if the subset of words contains only alphabets then return the words separated with a space
                    if (len([str for str in word.split("-") if not str.isalpha()]) == 0):
                         return re.sub(r'-'," ",word)
                    #else return an empty string as the words might contain other characters or numbers indicating that it might be a product id
                    return ""
                else:
                    return ""

    # this function preprocess a raw string (caption) and returns the preprocessed string (caption) in lower case
    def __preprocess_str(self, caption):
        caption=re.sub(r'w\/', "with", caption) #replace w/ with with
        caption=re.sub(r'\&', "and", caption) #replace & with and
        caption=re.sub(r'[,|(|)|\/]', " ", caption) #eliminate breackets
        caption=" ".join([self.__preprocess_word(word) for word in caption.split() ]) #preprocess each word in the caption and combine them to a list
        #caption=" ".join([word for word in caption.split() if word.isalpha() ])
        #caption=re.sub(r'\s+[A-z|a-z]+(\-|\d+)[A-z|a-z|0-9|-]*', " ", caption)
        #caption=re.sub(r'[A-Z|a-z]+(-)*[A-Z|a-z]*\d+\w*', "", caption)
        #caption=re.sub(r'[/|\-|\|]', " ", caption)
        #caption=re.sub(r'\s*\d+(\.\d+)*\s*"*\s*(x|X)\d+(\.\d+)*\s*"*\s*(x|X)\d+(\.\d+)*\s*"*\s*', " ", caption)
        caption=re.sub(r'\s+(Inch|in|W|H|D)\s+', " ", caption) #eliminate Inch and other dimension indicators
        caption=re.sub(r'\s+x\s+', " ", caption)
        return caption.lower()

    #preprocess each caption and store the preprocessed caption in the dataframe
    def preprocess_caption(self):
        for i in range(0,self.len_traindir):
            self.outputdf= self.outputdf.append({'filename':self.inputdf['filename'][i], 'caption':self.__preprocess_str(self.inputdf['caption'][i])}, ignore_index=True)
        #print(self.outputdf)
        self.remove_furniturecompanies_fromcaption()

    #returns dictonary of all the words in the preprocessed captions data frame
    def dict_words(self):
        dict_all_words={}
        for i in range(0, self.len_traindir):
            caption = self.outputdf['caption'][i]
            words= caption.split()
            for word in words:
                if word in dict_all_words.keys():
                    dict_all_words[word]= dict_all_words[word] + 1
                else:
                    dict_all_words[word]= 1
        return dict_all_words

    #name of the furniture companies, appear in the start of the caption
    #remove names of the furniture companies from the start of the captions
    def remove_furniturecompanies_fromcaption(self):
        capdict={}
        len_furniture_name=3 #furniture names upto 3 words
        #go through each training caption and extract upto the first len_furniture_name dict_words
        #and store it in the capdict dictonary along with their frequency of occurence
        for index in range(0,self.len_traindir):
            caption=self.outputdf['caption'][index]
            for i in range(0,len_furniture_name):
                caption = caption.lower()
                firstwords= caption.split()[0:i]
                firstwords = " ".join(firstwords)
                if firstwords in capdict.keys():
                        capdict[firstwords]=capdict[firstwords]+1
                else:
                        capdict[firstwords]=1

        captlist=[]
        #identify a possible list of furniture names and store it in captlist
        for i in range(0,self.len_traindir):
            caption = self.outputdf['caption'][i]
            caption=caption.lower()
            print(caption)
            firstwords= caption.split()
            max_val=capdict[" ".join(firstwords[0:0])]
            captindex=0
            for j in range(1, len_furniture_name):
                currcaption = " ".join(firstwords[0:j])
                #print(currcaption)
                if capdict[currcaption] < max_val and capdict[currcaption]>2:
                    max_val=capdict[currcaption]
                    captindex=j
                    captlist.append(" ".join(firstwords[0:captindex]))
            captlist.append(" ".join(firstwords[0:captindex]))

        #combine the furniture names with the existing list of brands and
        #delete those from the dictoinary
        set_captlist=set(captlist+brands)
        print(set_captlist)
        newcol=[]
        for i in range(0,self.outputdf.shape[0]):
            assigned=False
            for j in range(5,0,-1):
                #print(" ".join(self.outputdf['caption'][i].split()[0:j]))
                if " ".join(self.outputdf['caption'][i].split()[0:j]) in set_captlist:
                    newcol.append(" ".join(self.outputdf['caption'][i].split()[j+1:]))
                    assigned=True
                    break
            if not assigned :
                newcol.append(self.outputdf['caption'][i])
        self.outputdf['caption_old']=self.inputdf['caption']
        self.outputdf['caption_new']=newcol
        self.outputdf.to_csv(os.pardir+"Amazon_furniture_editedcaptions_2.csv")

    #compute the vocabulary length
    def compute_the_vocabulary():
        vocabulary=set()
        for i in range(self.outputdf.shape[0]):
            ls = [word for word in newdf['caption_new'][i].split() if not (len(word) == 1) or word == 'l' ]
            self.outputdf['caption_new'][i]=[word for word in newdf['caption_new'][i].split() if not (len(word) == 1) or word == 'l' ]
            for word in ls:
                    vocabulary.add(word)
        return len(vocabulary)

def main():
        DataPreprocess = DataPreprocessing()
        DataPreprocess.display_image_caption(2)
        DataPreprocess.preprocess_caption()


if __name__ == "__main__":
        main()

"""
{'', 'signature', 'sterling', 'folding convertible', 'divano roma furniture', 'zhlj', 'serta geneva', 'limari', 'modern sectional', 'american eagle furniture', 'vig furniture', 'ethan', 'stone', 'homegear', 'boss', 'naomi', 'esofastore classic', 'sealy victor', 'funrelax', 'sofa bed', 'lmz nordic', 'dhp', 'pemberly', 'xiaosunsun', 'great', 'poshbin wilson', 'vidaxl convertible', 'kardiel cumulus', 'office', 'pasargad', 'sawyer', 'fabric', 'elegant', 'lmz', 'brand', 'us pride furniture', 'brown', 'home source', 'baxton', 'poshbin', 'burrow', 'three', 'unfade', 'olee', 'epic furnishings', 'simmons', 'tov', 'upholstered', 'sofa trendz', 'mainstays apartment', 'ghp', 'gdfda high', 'campaign', 'whiteline', 'rocket', 'corliving club', 'loveseat sofa', 'cortesi', 'sunset trading', 'aimcae', 'esofastore casual', 'coaster sofa', 'serta deep', 'rivet revolveserta', 'oliver', 'gdf', 'ff', 'poundex', 'gdfstudio mckinley', 'bowery hill', 'ghp capacity', 'carolina', 'nature utility', 'sofas couches', 'wrea', 'lounge chair', 'velago rossini', 'velago', 'coaster home', 'single', 'vidaxl modern', 'furinno simply', 'legend', 'marceooselm', '24 7 shop at home', 'modern large', 'divano', 'wsn', 'mamasam convertible', 'meridian furniture', 'dnnal sofas', 'furniture of america', 'great deal furniture', 'container furniture direct', 'abbyson', 'tommy', 'fflsdr', 'modern bonded', 'istikbal elita', 'morrisofa william', 'flash furniture', 'klaussner', 'white', 'polaris mini', 'tk classics', 'jean', 'design', 'oliver pierce', 'casual style', 'simpli', 'furniture for', 'emerald', 'kmp', 'large inch', 'acme sofa', 'roundhill', 'hugo', 'handy', 'zhihuitong', 'ashley', 'velvet inch', 'david', 'modern living', 'morrisofa everly', 'modern style', 'zinus ricardo', 'lazy', 'xvtyxsxio', 'american eagle', 'best master', 'lounge malibu', 'catalina', 'signature design by ashley', 'jresboen', 'xiao', 'global', 'porter', 'divani', 'brika', 'christopher', 'infini', 'rivet uptown', 'sofas', 'modern', 'kanizz deluxe', 'danxee', 'sealy anson', 'pemberly row', 'we', 'mewmewcat', 'harlow', 'serta astoria', 'fs lazy', 'work', 'coaster fabric', 'ilovo', 'velago ollon', 'classic', 'zuo varietal', 'home styles', 'vegas futon', 'boca rattan', 'convertible', 'modern contemporary', 'ac', 'zuo', 'tribecca', 'yyh single', 'modway resolute', 'black', 'milan sofa', 'lazy sofa', 'large', 'classic brandschristopher knight home', 'dg casa', 'polaris', 'bali', 'stylistics hunter', 'pangea', 'seaphy', 'acme kiva', 'wrea three', 'homelegance resonance', 'milan', 'glory', 'rivet frederick', 'rattan', 'taylor', 'european furniture', 'loungie', 'chelsea', 'modway', 'loveseat chaise', 'wsn corner', 'poundex sofas', 'universal', 'dnnal faux', 'classic linen', 'dandd', 'foldable dual', 'dnnal fabric', 'kotag', 'brika home', 'major', 'greatime top', 'esofastore modern', 'fdw', 'jandm', 'global supplies', 'convertible sectional', 'lisbona', 'gdfstudio', 'furniture sofa', 'simmons upholstery', 'eurotech', 'homeroots furniture', 'home', 'indoor', 'tidyard', 'charles ashton', 'zhlj solid', 'duobed', 'xavieradoherty', 'lmz lazy', 'aodailih', 'benzara flannelette', 'steve', 'folding lazy', 'lesro', 'iconic', 'flip', 'alera qub', 'extra', 'home life', 'monarch', 'leather', 'festnight faxu', 'lane', 'bowery', 'global furniture usa', 'seatcraft', 'dorel', 'serta', 'hwy', 'rattan living', 'tly cotton karlstad', 'larson', 'acme furniture', 'festnight modern', 'lounge', 'dark', 'honbay sofa', 'starsun', 'kardiel monroe', 'homeroots', 'pearington', 'ids', 'coaster roy', 'wrea fabric', 'thomas', 'elle', 'south cone home', 'velvet', 'iconic home', 'flash', 'pulaski', 'yskwa', 'j and m furniture', 'coja', 'domesis granada', 'aynefy', 'gray velvet', 'benjara', 'mercana', 'franklin', 'poly', 'nrthtri', 'glory furniture', 'cambridge contemporary', 'dnnal', 'zentique', 'merax', 'poundex black', 'acme chantelle', 'container furniture', 'benjara leather', 'rivet sloane', 'jinpengran', 'esofastore', 'vidaxl chesterfield', 'wurink', 'zoubiag', 'vvr homes', 'benchcraft charenton', 'serta sierra', 'rattan and', 'zhlj lazy', 'offex', 'bestmassage', 'sofa side', 'truly', 'rivet damien', 'beam', 'domesis scooped', 'xhlxhl', 'best choice products', 'honbay convertible', 'mainstays', 'sofa furniture', 'upholstered lazy', 'thaweesuk', 'istikbal fantasy', 'gold', 'jennifer taylor home', 'lexicon', 'furniture of', 'tosh', 'best choice', 'benjara contemporary', 'modern extra', 'sunpan', 'blackjack', 'cambridge', 'victoria', 'modhaus', 'sofa leather', 'corliving', 'mandycng', 'modern two', 'kure', 'foldable', 'apt2b', 'lexicon barberton', 'futon', 'knocbel', 'esofastore living', 'south cone', 'ff lazy', 'greatime', 'madison', 'camper', 'jamie', 'beyan sb', 'alera', 'gy', 'irene', 'signature design', 'hooker', 'ikea', 'crafters', 'armen living', 'contemporary home', 'kardiel woodrow', 'fillmore', 'auroraelight', 'stylistics', 'domesis', 'acanva collection', 'morrisofa', 'homelegance casoria', 'dwell', 'fitnessclub', 'redde', 'serta rane collection', 'kemanduo', 'acme zuriel', 'america luxury sofa', 'wsn fabric', 'convertible sofa', 'multifunctional', 'dreamseat', 'skb', 'benjara benzara', 'rivet cove', 'nhi', 'homestyles', 'kings', 'made', 'aimcae convertible', 'sunset', 'zinus josh', 'knocbel upholstered', 'south', 'coaster velvet', 'dhp andora', 'container direct', 'comfortmax', 'xleve', 'homelegance bastrop', 'best', 'oadeer home', 'fdw recliner', 'zinus juan', 'tevahome', 'qxx', 'loveseat', 'ihpaper', 'marceooselm leisure', 'golden', 'modern franco', 'teenage', 'inside', 'kingway', 'giantex', 'homelegance center', 'powell', 'hydeline', 'nanapluz', 'mikewei', 'serta mason', 'vig', 'folding sofa', 'scm', 'canditree modern', 'grey', 'acme vendome', 'south shore', 'christies', 'homeroots upholstery', 'nature', 'dhp emily', 'coaster tufted', 'oadeer', 'recpro', 'baja', 'benjara leatherette', 'zinus benton', 'kardiel', 'stone and beam', 'posh', 'design tree', 'sealy montreal', 'sofa sleeper', 'mainstays sleeper', 'harperandbright', 'acanva', 'adjustable floor', 'vegas', 'ethan allen', 'chelsea home', 'night', 'jennifer', 'casual folding', 'vintage', 'adjustable', 'blackjack furniture', 'serta palisades', 'la', 'luton sectional', 'my', 'homelegance taye', 'belmont', 'global furniture', 'rivet revolve', 'l', 'yskwa double', 'ofm', 'kanizz contemporary', 'furniture', 'serta rta palisades collection', 'pulaski upholstered', 'navy', 'sleeper', 'modway empress', 'lifestyle', 'fold', 'new', 'garden', 'rivet north', 'poundex bobkona', 'furinno', 'convertible recliner', 'charles', 'massage', 'kanizz', 'modern glam', 'us', 'corliving lea', 'noble', 'scdxj', 'eqsalon', 'loveseat modern', 'finch', 'pearington macon', 'upholstered inch', 'belagio', 'esofastore sectional', 'lazzaro leather', 'italian', 'cozinest pu', 'lilola', 'us pride', 'brentwood', 'contemporary', 'logan', 'us furnishing', 'burrow nomad', 'mamasam', 'armen', 'lucky', 'contemporary three', 'sofa', 'container', 'armrest', 'serta rta', 'acme', 'folding', 'coaster quinn', 'mid', 'leather sofa', 'modern leather', 'edloe', 'golden coast', 'funrelax sectional', 'meridian', 'homelegance aram', 'modern sofa', 'adianna', 'sofas two', 'harper', 'casual', 'zinus', 'zuri', 'pearington multifunctional', 'xiaosunsun customized', 'samuel', 'vidaxl sofa', 'tidyard modern', 'modern black', 'gray sofa', 'new classic', 'classic large', 'classic two', 'chlfsfd', 'new felton', 'coaster', 'madison home', 'zuo jonkoping', 'jnm', 'gdfda', 'fandf', 'joanna', 'ilovo double', 'festnight', 'poundex upholstered', 'serta upholstery', 'lcb', 'ms', 'leather sectional', 'young', 'zinus jackie', 'yyh', 'adianna sofa', 'ashley furniture', 'modway delve', 'gray', 'offex contemporary', 'brown faux', 't', 'hawthorne', 'esther', 'classic and', 'krei', 'genuine', 'greyson', 'american furniture', 'benzara', 'benzara button', 'fat', 'kathy', 'agata', 'classic living', 'contemporary modern', 'zdnals', 'beyan', 'istikbal', 'sectional', 'acme ceasar', 'dhingm', 'novogratz', 'american', 'inspire', 'ashley furniture signature design', 'leisuremod', 'wood', 'ravenna', 'q', 'button', 'mechali', 'casa andrea', 'beverly', 'classic chesterfield', 'cozinest', 'yoyi', 'lexicon eli', 'baxton studio', 'vita', 'daonanba', 'truemodern', 'flash furniture benchcraft maier', 'living', 'european', 'j', 'klaussner neyland', 'mewmewcat sofa', 'uniters', 'mademai', 'nsk', 'blue', 'upholstered leather', 'artum', 'mission', 'safavieh', 'rivet', 'homes: inside out', 'benzara sectional', 'casa mare', 'serta copenhagen', 'lazy couch', 'futon sofa', 'truemodern jackson', 'giantex futon', 'vig contemporary', 'artdeco', 'sofa for', 'zinus lauren', 'luton', 'morrisofa cameron', 'homelegance', 'benchcraft brindon', 'beige', 'futon sleeper', 'benchcraft', 'blue velvet', 'zhlj zero', 'ikea cover', 'zhihuitong modern', 'fs', 'zhlj tatami', 'lazzaro', 'vidaxl', 'lifestyle solutions', 'lmz comfortable', 'homelegance welty', 'newport', 'mecor', 'diamond', 'homeroots x', 'sealy', 'modway valour', 'casa', 'chic', 'giantex fold', 'nature full', 'sectional sofa', 'aycp', 'rabinyod', 'housel', 'scott', 'xavieradoherty bed', 'limari home', 'hydt', 'sunpan modern', 'honbay', 'homelegance belmont', 'progressive', 'kuber', 'sandy', 'dreamhank', 'a', 'yaheetech', 'acanva chesterfield', 'canditree', 'coaster home furnishings', 'zinus sunny', 'at', 'tufted', 'tidyard leather', 'office star', 'inspired', 'l shape', 'littletonhome', 'baja convert', 'fun', 'case', 'erssst', 'sleeper sofa', 'tufted sofa'}

"""
