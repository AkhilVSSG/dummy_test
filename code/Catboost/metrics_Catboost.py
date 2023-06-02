#importing the required modules

import json
import plotly.graph_objects as px_1
import plotly.express as px
import numpy as np
import pandas as pd




def get_act_data(f_name):
    '''
    to get the actual data of the symptoms and their respective labels
    
    input:
        f_name(str) : store the name of the file for which actual data is required

    output:
        checklist(list) : it stores the actual data in form of the symptom and its repective tags

    '''
    with open("../data/"+f_name+".json" ,'r', encoding='utf-8') as read_file:
        f = json.loads("[" +read_file.read().replace("}\n{", "},\n{") + "]")
        checklist=[]
        for i in f:
            checklist.append([i["title"],i["tags"]])
    return checklist

def get_pra_data(f_name,model):
    '''
    to get the predicted data of the symptoms and their respective labels
    
    input:
        f_name(str) : store the name of the file for which actual data is required
        model(str): stores the name of the model from which the predicted data is created

    output:
        resp(list) : it stores the predicted data in form of the symptom and its repective tags with thier predicted probabilities

    '''

    with open("../data/"+model+"/"+f_name+"_infer_"+model+".txt","r") as file: 
        resp=[]
        for i in file:
            x=json.loads(i)
            resp.append([x["title"],[[j[0],round(j[1],4)] for j in sorted(x["predict"],reverse=True,key=lambda Z: Z[1])]])
    return resp

def model_metrics(model):
    '''
    to calculate the precision @k and accuracy of the test and val datasets w.r.t different thresholds
    
    input:
        model(str): stores the name of the model used in the pipeline

   
    '''    
    def get_precision(actual_data,predicted_data,category,model):
            ''''
            to get precision@k for the respective data

            input:
                actual_data (list) : it stores the actual data in form of the symptom and its repective tags
                predicted_data (list) : it stores the predicted data in form of the symptom and its repective tags with thier predicted probabilities
                category(str) : store the name of the file for which precision is calculated
                model (str) : stores the name of the model from which the predicted data is created

            '''
            

            def get_freq(data):
                '''
                to get frequency of the number of the labels preset in the dataset
                
                input:
                    data(list): stores the data to which the frequency must be calculated
                
                ouput:
                    freq(dict) : stores the frequencies of the number of the labels preset in the data
                '''
                freq={}
                for i in data:
                    x=len(i[1])
                    freq[x]=freq.get(x,0)+1
                return freq
        
            ans=[]
            predicted,actual=[],[]
            freq_data = get_freq(actual_data)
            pres=list(freq_data.keys()) # gets the list of how many different length of labels are oresnt in the dataset

            for precision in pres:
                final_ans=0
                for i in range(len(predicted_data)):
                    temp_ans=0

                    # checking weather the labels upto a certain precision in the predicted data are presnt in the actual labels are not
                    
                    for each in predicted_data[i][1][:precision]:
                        if each[0] in actual_data[i][1]:
                            temp_ans+=1
                    final_ans+=(temp_ans/precision)
                predicted.append(final_ans*100/len(actual_data))
                ans.append("precision@"+str(precision)+" = "+str(final_ans*100/len(actual_data)))
            # print(*ans,sep='\n')

            # this is for the actual precison at a particular k

            for pre in pres:
                f_ans=0
                for i in freq_data:
                    f_ans+=freq_data[i]*min(i,pre)/pre
                actual.append(f_ans*100/len(actual_data))

            # a bar grpah with actual and predicted precision @k

            X=["P@"+str(k) for k in pres]
            # print(actual,predicted)
            plot = px_1.Figure(data=[px_1.Bar(
                name = 'Theoretical',
                x = X,
                y = actual
               ),
               px_1.Bar(
                name = 'Practical',
                x = X,
                y = predicted
               ),
            
            ])
            plot.update_layout(
                title="Precision @ k "+category+" "+model,
                xaxis_title="Precision @ k",
                yaxis_title="Percentage(%)",
            
            )

            plot.write_html("../static/plots/"+model+"/Precision@k_"+category+"_"+model+".html")   

        
    X,Y,cat=[],[],[]

    for name in ["test","val"]:
        actual_data,predicted_data=get_act_data(name),get_pra_data(name,model)

        get_precision(actual_data,predicted_data,name,model)

        accuracy = []
    
        for j in np.arange(0,1 , 0.0001):
            j=round(j,4)
            acc=0
            for i in range(len(predicted_data)):
                temp=[]

                # checking weather the threshold of the current prabability of a diagnosis is more not not if it more then it is considered

                for each in predicted_data[i][1]:
                    if each[1]> j:
                        temp.append(each[0])

                if sorted(actual_data[i][1])==sorted(temp): acc+=1
            
            accuracy.append([j,acc*100/len(actual_data)])

        # converting the accuracies of a particular category into a csv for a line graph

        for each in accuracy:
            X.append(each[0])
            Y.append(each[1])
            cat.append(name)
    
    dict={"Threshold":X,"Accuracy":Y,"Category":cat}

    df = pd.DataFrame(dict) 

    fig = px.line(df, x='Threshold', y='Accuracy', color='Category')
    fig.update_layout(
        title="Val vs Test Accuracy w.r.t Threshold "+model.upper() ,
    )
    fig.show()
    fig.write_html("../static/plots/"+model+"/VAL_vs_TEST_("+model.upper()+").html")    
