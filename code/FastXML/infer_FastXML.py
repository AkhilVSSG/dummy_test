#importing the required modules

import os


def infer_fastxml(f_name,model):
    '''
    used for the inference of the test and the val dataset with the trained FastXML model
    
    input:
        f_name (str) : stores the file name to which the model should perform inference
        model(str) : stores the name of the model which is trained
    '''
    
    file_="../data/FastXML/"+f_name+"_infer_"+model+".txt"

    # in the below step we are activating the virtual environment for th FastXML execution and then running the infer command and storing the result in a txt file 
    
    os.system("fxml.py ../static/models/FastXML/Fastxml.model ../data/"+f_name+".json inference --score >"+file_)   
