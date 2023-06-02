#importing the required modules

import os




def train_fastxml():
    '''used for the train the FastXML model with the help of the train dataset'''

    # in the below step we are activating the virtual environment for th FastXML execution and then running the train command and storing the result in a txt file 

    os.system("fxml.py ../static/models/FastXML/Fastxml.model ../data/train.json --verbose train --iters 2000 --trees 40 --label-weight propensity --alpha 1e-4 --leaf-classifiers")      
    