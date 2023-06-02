import json
import numpy as np
import pandas as pd
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


def clean_text(text):
    '''
        Make text lowercase
    '''
    text=' '.join([i.strip() for i in text.strip("[]").replace("'","").split(',')])
    return text

def preprocess_data(text):
    text = clean_text(text)  
    stop_words = stopwords.words('english')
    stemmer    = nltk.SnowballStemmer("english")
    text = ' '.join(word for word in text.split() if word not in stop_words)    # Remove stopwords
    text = ' '.join(stemmer.stem(word) for word in text.split())                # Stemm all the words in the sentence
    return text


def data_preprocess():
    global train_div,test_div,val_div

    data = pd.read_csv('../data/data.csv') #load the csv data to be preproccesed 
    df = pd.DataFrame(data)

    


    df1=df[['Provisional_diagnosis','combined_symptoms']]
    
    df1['clean_combined_symptoms'] = df1['combined_symptoms'].apply(preprocess_data)

    # df1['title']=df1['Symptoms']+' '+df1['ProvisionalDiagnosis']
    df1=df1.drop(['combined_symptoms'],axis=1)
    data=[]
    unique_med=dict()
    for i,j in df1.itertuples(index=False):

        med=i.split('and') #splitting the varioud lables

        tags=[i.strip().lower() for i in med]

        for tag in tags:
            unique_med[tag]=unique_med.get(tag,0)+1 #storing all the uniquely present tags
    
        title= ''.join([word for word in j.lower() if word.isalnum() or word==' '])

        data.append({"title":title,"tags":tags})


    # np.random.seed(0)

    # shuffling the data and splitting the gievn data into test,  train and val

    random.shuffle(data)
    np.random.seed(0)
    train,val,test=np.split(data,[int(train_div*len(data)),int((1-val_div)*len(data))])


    for i in ["train","test","val"]:
        x=train if i=="train" else test if i=="test" else val
        with open("../data/"+i+".json","w") as w_file:
            for j in x:
                json.dump(j,w_file)
                w_file.write("\n")

    unique_med=sorted(unique_med.items(),reverse=True,key= lambda X:X[1])

    with open("../data/unique_lables.txt","w") as w_file:
        for i in unique_med:
            w_file.write(i[0]+"\n")

def main():
    global train_div,test_div,val_div

    # below we are taking all the values for the train/test/val split which are user defined and will be used in training the model


    with open("../data/config.json" ,'r', encoding='utf-8') as read_file:
        model_data = json.loads("[" +read_file.read().replace("}\n{", "},\n{") + "]")[0]

    train_div,test_div,val_div=model_data.get('train/test/val')

    data_preprocess()

if __name__ == "__main__":
    main()