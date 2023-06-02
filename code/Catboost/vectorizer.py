#importing the required modules

import pickle
from sklearn.feature_extraction.text import CountVectorizer
import json
# import os


lables=dict()
medicines=dict()#to store the unique medicines in a dictionary 
vect = CountVectorizer() #initilizing CountVectorizer

def dev_vectorizer(model_type):
    '''
    train and trainform the given symptom data into vectors using CountVectorizer
    input:
        model_type(str): stores the name of the model from which the predicted data is created
    
    output:
        lables(dict) : to store the unique medicines in a dictionary in the form of medicine and index as a key-value pair 
        medicines (dict) : to store the unique medicines in a dictionary in the form of index and medicine as a key-value pair 


    '''

    global lables,medicines,vect
    # print(os.getcwd())
    with open("../data/unique_lables.txt",'r') as fun:
        num=0
        
        for i in fun:

            lables[i.strip()]=num
            medicines[num]=i.strip()
            num+=1

    text=[]
    for i in ['train']:
        with open("../data/"+i+".json",'r') as file:
            js = json.loads("[" +file.read().replace("}\n{", "},\n{") + "]")
            for j in js: text.append(j['title'])
    vect.fit(text)
    pickle.dump(vect, open("../static/models/"+model_type+"/vectorizer_catboost", "wb"))

    return lables,medicines
