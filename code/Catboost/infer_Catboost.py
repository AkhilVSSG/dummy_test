#importing the required modules

import json
import pickle



def infer_catboost(model_type,lables,medicines,iter,depth):
        '''
            for inference of the test and the val dataset for the trained model
            
            input:
                iter (int) : it stores the number of iteration that the catboost mode had trained 
                depth (int) : it stores the maximum depth to which a tree could grow while trianing the catboost model
                lables(dict) : to store the unique medicines in a dictionary in the form of medicine and index as a key-value pair 
                medicines (dict) : to store the unique medicines in a dictionary in the form of index and medicine as a key-value pair 
                model_type(str): stores the name of the model from which the predicted data is created

            '''
        
        # loading the count vectorizer which was fit with the train dataset

        vect=pickle.load(open("../static/models/"+model_type+"/vectorizer_catboost", "rb"))

        # loading the cCatboost mode which was fit with the train dataset


        path="../static/models/"+model_type+"/catboost_"+str(iter)+"_iter_"+str(depth)+"_depth.pickle"
        with open(path, "rb") as input_file:
            ovr_1 = pickle.load(input_file)
        
        # encoding the test and val datasets with help of count vectorizer

        for name in ['test','val']:
            X,Y=[],[]
            q_name=[]
            with open("../data/"+name+".json",'r') as file:
                js = json.loads("[" +file.read().replace("}\n{", "},\n{") + "]")
                x,y=[],[]
                for j in js:
                    x.append(j['title'])
                    y.append([lables[z]+1 for z in j['tags']])
                temp=vect.transform(x)
                X,y=temp.toarray(),y
                q_name=x
   
            x=ovr_1.predict_proba(X)

            # storing the infered data into a txt file with probabilities of various diagnosis

            with open("../data/"+model_type+"/"+name+"_infer_"+model_type+".txt","w") as wt_f:
                for i in range(len(x)):
                    temp=[]
                    for j in range(len(x[i])):
                        temp.append([j,x[i][j]])
                    temp=sorted(temp,key = lambda Z : Z[1],reverse = True)[:10]
                    json.dump({"title":q_name[i],"predict":[[medicines[i[0]],i[1]] for i in temp]},wt_f)
                    wt_f.write("\n")
