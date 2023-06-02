# Pipeline_Multi_label

This pipeline is used for Multi label classification which has 3 models namely FastXML , Catboost(with OneVsRest classification) , and Bert.

the directory roadmap is:
```
 pipeline
    code
        Bert
            convert_to_csv.py
            metrics_Bert.py
            train_Bert.py
        Catboost
            infer_Catboost.py
            metrics_Catboost.py
            train_Caboost.py
            vectorizer.py
        FastXML
            infer_FastXML.py
            metrics_FastXML.py
            train_FastXML.py
        fastxml(environment)
        data_preprocess.py
        pipeline.py
    data
        config.json
    static
        plots (saves the plots for the respective models)
        models (saves the trained model)
  ```

1. First clone the repository

    ```sh
    git clone https://github.com/AkhilVSSG/Multi_label_Pipeline.git
    ```
    then,
    ```sh
    cd Multi_label_Pipeline/pipeline/
    ```

2. To start this pipeline first thing is to create a conda environment and install all the dependencies in the requirement.txt
	- Build a conda environment
        ```sh
        conda create --name env_name
        ```
	- Activate the environment
        ```sh
        conda activate env_name
        ```
    - Install all the dependencies in the requirements.txt
        ```sh
        pip install -r requirements.txt (Python 2)
        pip3 install -r requirements.txt (Python 3)
        ```

3. Upload the raw data into the data directory.

4. Then we must run the data_preprocess.py you can edit the py file according to the columns you require and other requirements
	- After you have changed the data_preprocess.py file according to your requirements then run the below command
		* First go to code directory
		    ```
            cd code/
            ```
        * Make changes to the config.json file in the data directory
       
        * Then run this command to run the data_preprocess file
           ```
           python data_preprocess.py
           ```
5. You can see you will have train.json,test.json,val.json and unique_labels.txt created

6. Then we must run the pipeline file it will take data from the config file and start training
        ```
        python piepline.py
        ```
7. After the execution of the py file we can see various files created that is we can see
	- Inference files for test and val created in the respective model section in the data folder
    - Trained model file in the models section which is in static folder
    - Plots for precision @ k for test and val datasets and the accuracy vs threshold plot in the plots section in the static folder

8. You can change the config file and train different models for the given data.

YOU ARE GOOD TO GO!!!

Personalization according to the user:
1. You can change the number of iterations and number of trees used by the FastXML model by going into the train_FastXML in FastXML in code directory and do the necessary changes

2. In the same way you can change the number of epochs in which a Bert model mu train which is available in train_Bert in Bert in code directory

3. You can even change the number of iterations and depth of a tree in Catboost by changing their values in the train_Caboost ni Catboost in code directory

4. all the models like Catboost and Bert are using their primary encoding methods i.e. CountVectorizer and auto tokenizer using roberta-base you can even change those and train your model. Those can be changed in vectorizer.py in Catboost and train_Bert.py in Bert respectively


Note: 
1. if you are having any trouble using FastXML please refer https://github.com/Refefer/fastxml
2. for Bert refer https://www.kaggle.com/code/debarshichanda/bert-multi-label-text-classification/notebook

