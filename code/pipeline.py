#importing the required modules


import json
import os

#importing the required function to perform the Bert model

from Bert.convert_to_csv import convert_to_csv
from Bert.train_Bert import train_bert
from Bert.metrics_Bert import model_metrics as Bert_metrics

#importing the required function to perform the Catboost model


from Catboost.vectorizer import dev_vectorizer
from Catboost.train_Catboost import train_catboost
from Catboost.infer_Catboost import infer_catboost
from Catboost.metrics_Catboost import model_metrics as Catboost_metrics

#importing the required function to perform the FastXML model


from FastXML.train_FastXML import train_fastxml
from FastXML.infer_FastXML import infer_fastxml
from FastXML.metrics_Fastxml import model_metrics as Fastxml_metrics

model_type=''

def dev_bert():
    '''performing training, inference and metric calculation of the Bert model'''

    global model_type

    no_of_labels,medicines=convert_to_csv(model_type)
    train_bert(model_type,medicines,no_of_labels)
    Bert_metrics(model_type)

def dev_catboost():
    '''performing training, inference and metric calculation of the Catboost model'''


    global model_type

    lables,medicines= dev_vectorizer(model_type)
    iterations,depth=train_catboost(model_type,lables)
    infer_catboost(model_type,lables,medicines,iterations,depth)
    Catboost_metrics(model_type)    

def dev_fastxml():
    '''performing training, inference and metric calculation of the FastXML model'''

    global model_type

    train_fastxml()
    infer_fastxml("test",model_type)
    infer_fastxml("val",model_type)
    Fastxml_metrics(model_type)



def main():

    global model_type

    # below we are taking all the values for the model type embeding etc which are user defined and will be used in training the model

    with open("../data/config.json" ,'r', encoding='utf-8') as read_file:
        model_data = json.loads("[" +read_file.read().replace("}\n{", "},\n{") + "]")[0]
    model_type=model_data.get('model_type')
    embeding_type=model_data.get('embeding')

    os.chdir("..")
    path = os.getcwd()

    os.makedirs(path+"/data/"+model_type, exist_ok = True)

    os.makedirs(path+"/static/plots/"+model_type, exist_ok = True)
    os.makedirs(path+"/static/models/"+model_type, exist_ok = True)

    os.chdir("code/")

    if model_type=="FastXML":
        dev_fastxml()
    if model_type=="Catboost":
        dev_catboost()
    if model_type=="Bert":
        dev_bert()



if __name__ == "__main__":
    main()
