#importing the required modules

import json
import pandas as pd
import numpy as np
from sklearn import metrics


import torch
torch.cuda.empty_cache()
from torch.utils.data import Dataset, DataLoader
from transformers import  AutoTokenizer, AutoModel, AdamW

import os
import warnings
warnings.filterwarnings('ignore')


def train_bert(model_type,med,num_lables):
        '''
        for training the train dataset with the mention model
            
        input:
            num_lables(int) : it stores the number of unique lables present in the data
            med (dictionary) : it stores the medicine and its respective index as value and key pair respectively
            model_type(str): stores the name of the model from which the predicted data is created

        '''

        def infer(loader,cat):
            '''
            for inference of the test and the val dataset for the trained model
            
            input:
                loader(DataLoader from torch.utils) : it stores the preprocessed dataset using torch modules 
                cat (str) : it stores the catogory to which the model must infer the data
               
            '''

            q_name=[] #to store the symptoms

            with open("../data/"+cat+".json",'r') as file:
                js = json.loads("[" +file.read().replace("}\n{", "},\n{") + "]")
                x=[]
                for j in js:
                    x.append(j['title'])        
                q_name=x
            
            
            model.load_state_dict(torch.load('../static/models/'+model_type+'/bert_model_'+str(EPOCHS)+'_epochs.bin'))


            def validation():
                model.eval()
                fin_outputs=[]
                with torch.no_grad():
                    for _, data in enumerate(loader, 0):
                        ids = data['ids'].to(device, dtype = torch.long)
                        mask = data['mask'].to(device, dtype = torch.long)
                        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                        targets = data['targets'].to(device, dtype = torch.float)
                        outputs = model(ids, mask, token_type_ids)
                        fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                return fin_outputs


        
            outputs = validation()
           
            dict_med={}
            
            # storing the infered data into a txt file with probabilities of various diagnosis

            with open("../data/"+model_type+"/"+cat+"_infer_"+model_type+".txt","w") as wt_f:
                for j in range(len(outputs)):
                    for i in range(len(outputs[j])):
                        dict_med[i]=outputs[j][i] #storing the output probabilities along with the diagnosis

                    sorted_med=sorted(list(dict_med.items()), reverse=True,key= lambda  x:x[1])[:10]

                    medicines=[]
                    for i in sorted_med:
                        x=list(i)
                        
                        medicines.append([med[x[0]],x[1]])

                    json.dump({"title":q_name[j],"predict":medicines},wt_f)
                    wt_f.write("\n")


        def get_max_length():
            '''to get the longest symptom for data preprocess step'''

            max_len_title=float('-inf')  
            for i in ['test','train','val']:
                with open('../data/'+i+".json",'r') as file:
                    js = json.loads("[" +file.read().replace("}\n{", "},\n{") + "]")
                    for j in range(len(js)): 
                        if len(js[j]['title'])>max_len_title:
                            max_len_title=len(js[j]['title'])

            return max_len_title


        

        pd.set_option("display.max_columns", None)

        # loading the csv

        df_train=pd.read_csv('../data/'+model_type+'/train.csv')
        df_test=pd.read_csv('../data/'+model_type+'/test.csv')
        df_val=pd.read_csv('../data/'+model_type+'/val.csv')


        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(device)

        # initializing various variables which is used in the process in the future

        MAX_LEN = get_max_length()
        TRAIN_BATCH_SIZE = 4
        VALID_BATCH_SIZE = 4
        TEST_BATCH_SIZE = 4
        EPOCHS = 5
        LEARNING_RATE = 2e-5
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')



        target_cols = [col for col in df_train.columns if col not in ['title']]


        # preprocessing the data and tokenizing it

        class BERTDataset(Dataset):
            def __init__(self, df, tokenizer, max_len):
                self.df = df
                self.max_len = max_len
                self.text = df.title
                self.tokenizer = tokenizer
                self.targets = df[target_cols].values
                
            def __len__(self):
                return len(self.df)
            
            def __getitem__(self, index):
                text = self.text[index]
                inputs = self.tokenizer.encode_plus(
                    text,
                    truncation=True,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    padding='max_length',
                    return_token_type_ids=True
                )
                ids = inputs['input_ids']
                mask = inputs['attention_mask']
                token_type_ids = inputs["token_type_ids"]
                
                return {
                    'ids': torch.tensor(ids, dtype=torch.long),
                    'mask': torch.tensor(mask, dtype=torch.long),
                    'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                    'targets': torch.tensor(self.targets[index], dtype=torch.float)
                }



        train_dataset = BERTDataset(df_train, tokenizer, MAX_LEN)
        valid_dataset = BERTDataset(df_val, tokenizer, MAX_LEN)
        test_dataset = BERTDataset(df_test, tokenizer, MAX_LEN)



        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, 
                                num_workers=4, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, 
                                num_workers=4, shuffle=False, pin_memory=True)
        
        test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, 
                                num_workers=4, shuffle=False, pin_memory=True)


        # loading a Bertclass

        class BERTClass(torch.nn.Module):
            def __init__(self):
                super(BERTClass, self).__init__()
                self.roberta = AutoModel.from_pretrained('roberta-base')
        #         self.l2 = torch.nn.Dropout(0.3)
                self.fc = torch.nn.Linear(768,num_lables)
            
            def forward(self, ids, mask, token_type_ids):
                _, features = self.roberta(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        #         output_2 = self.l2(output_1)
                output = self.fc(features)
                return output

        model = BERTClass()
        model.to(device);



        def loss_fn(outputs, targets):
            return torch.nn.BCEWithLogitsLoss()(outputs, targets)


        optimizer = AdamW(params =  model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)



        def train(epoch):
            model.train()
            for _,data in enumerate(train_loader, 0):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)

                outputs = model(ids, mask, token_type_ids)

                loss = loss_fn(outputs, targets)
                if _%500 == 0:
                    print(f'Epoch: {epoch}, Loss:  {loss.item()}')
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"


        for epoch in range(EPOCHS):
            train(epoch)

        torch.save(model.state_dict(), '../static/models/'+model_type+'/bert_model_'+str(EPOCHS)+'_epochs.bin')

        # calling infer function for infering the test and validation datasets

        infer(valid_loader,"val")
        infer(test_loader,"test")
