#importing the required modules

import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


num_lables=0 # to store number of unique lables

med=dict() #to store the unique medicines in a dictionary 

model_type=''

def convert_to_csv(model_type):
    '''
    convert the given json data into csv with the X column and one hot encodings of the labels
    
    input:
        model_type(str):gives us what type of model we are training
    
    output:
        num_lables(int) : it stores the number of unique lables present in the data
        med (dictionary) : it stores the medicine and its respective index as value and key pair respectively

    '''
    
    global num_lables,med


    lables=dict()
    with open("../data/unique_lables.txt",'r') as fun:

        for i in fun:
            z=i.strip()

            lables[z]=num_lables
            med[num_lables]=z

            num_lables+=1
    
    
    ques=[] #to store the text of the Symptom column
    medi=[] #to store the respective lables of the given symptom

    # converting the train,test and validation json datasets into csv

    for i in ['train','test','val']:
            with open('../data/'+i+".json",'r') as file:
                js = json.loads("[" +file.read().replace("}\n{", "},\n{") + "]")

                for j in range(len(js)): 

                    ques.append(js[j]["title"])
                    medi.append(js[j]["tags"])

    data_frame={"title":ques,"tags":medi}

    df = pd.DataFrame(data_frame)

    one_hot = MultiLabelBinarizer()
    y = pd.DataFrame(one_hot.fit_transform(df['tags']), columns=one_hot.classes_)
    df = df.join(y)
    df.drop('tags', inplace=True, axis=1)

    with open("../data/config.json" ,'r', encoding='utf-8') as read_file:
            model_data = json.loads("[" +read_file.read().replace("}\n{", "},\n{") + "]")[0]

    train_div,test_div,val_div=model_data.get('train/test/val')

    train,test,val=np.split(df,[int(train_div*len(df)),int((1-val_div)*len(df))])

    train.to_csv("../data/Bert/train.csv", index=False)
    test.to_csv("../data/Bert/test.csv", index=False)
    val.to_csv("../data/Bert/val.csv", index=False)
        
    return num_lables,med
