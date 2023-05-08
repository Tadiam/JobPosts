from transformers import  BertTokenizer 
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F
import torch 
from rake_nltk import Rake
import random
import matplotlib.pyplot as plt
import numpy as np
import logging 
from sklearn import metrics
from sentence_transformers import SentenceTransformer,util

from sklearn.metrics.pairwise import cosine_similarity

import os 
import sys 
path2 = os.path.abspath("../../JobPosts/")
sys.path.append(path2)
from eval.eval import Evaluate

class Job:
    """Job Class represents a single job entity with its description, index, score, and industry"""
    def __init__(self,description,index,industry):
        self.description=description
        self.index=index
        self.score=0
        self.industry=industry
        
class Job_Model:
    """Job_Model contains a model trained on specific jobs, using BertTokenizer and SentenceTransformer models.
        It contains a tokenizer, model, training and testing job list variables, and functions
        for pre-processing, testing, and training a model.
    """
    def __init__(self,  token='bert-base-nli-mean-tokens', ):

        self.model = SentenceTransformer(token)
        df=""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.jobs=[]
        self.industry=""
        self.train_jobs=[]
        self.test_jobs=[]
        self.length=0

    def __str__(self):
        """__str__ for Job_Model"""
        return f'Job Model for data set with path {self.path}'

    def __repr__(self):
        """__repr__ for Job_Model"""
        return f'Job Model, path (\'{self.path}\''

    def createJobList(self):
        """createJobList converts df into Job entities and stores them. Also calls seperate data"""
        count=0
        for x in self.df["Short Description"]:
            new_job=Job(description=x,index=count,industry=self.industry)
            count+=1
            self.jobs.append(new_job)
        self.seperateData()
        self.length=len(self.jobs)
       
    def seperateData(self):
        """seperateData creates training and testing lists of Job objects, with a 80/20 split"""
       
        current_industry=[self.jobs[x] for x in range(self.length-1,len(self.jobs))]
       
       
        this_industry_train=[]
        this_industry_test=[]
        stopper=len(current_industry)*.8
       
        y=0
        for x in current_industry:
            if y<=stopper:
                this_industry_train.append(x)
            else:
                this_industry_test.append(x)
            y+=1
        
        self.train_jobs.append(this_industry_train)
       
        self.test_jobs.append(this_industry_test)
       
    def trainModel(self):    
        """trainModel fine-tunes BERT model on dataset of resumes and job descriptions"""    
        logging.basicConfig(filename='train.log', encoding='utf-8', level=logging.INFO)
        length=0
        texts_against=[]
        length=0
        for s in self.train_jobs:
            for u in s:
                texts_against.append(u.description)
                length+=1
        for subset in self.train_jobs:
           new_array = []
           for i in self.train_jobs:
                    for j in i:
                        if j.description in subset:
                            new_array.append(1)
                        else:
                            new_array.append(0)
           train_labels = new_array
           label_tensor=torch.tensor(train_labels)
           for x in subset:
               
                train_texts = [x.description for l in range(length)]
                train_encodings = self.model.encode((train_texts))
                train_set_encoding=self.model.encode(texts_against)
                
                train_dataset = TensorDataset(torch.tensor(train_encodings), torch.tensor(train_set_encoding), label_tensor)

                train_dataloader = DataLoader(train_dataset, batch_size=10)

                optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

                self.model.train()
                for num in range(1):
                
                    for batch in train_dataloader:
                    
                        optimizer.zero_grad()
                       
                        train_embeddings,texts_against_embeddings, label = batch
                     
                        train_embeddings.requires_grad = True
                        texts_against_embeddings.requires_grad = True

                        label=label.float()
                        label.requires_grad = True
                        cosine_similarities = cosine_similarity(train_embeddings.detach().cpu().numpy(), texts_against_embeddings.detach().cpu().numpy())
                        
                        cosine_similarities = torch.from_numpy(cosine_similarities)
                
                        loss = torch.mean((cosine_similarities - label)**2)
                      
                        loss.backward()
                        
                        optimizer.step()
                
    def findClosestMatch(self,text):
        """findClosestMatch compares a given string to stored embeddings, and
        then sorts the jobs in ranked order according to their similarity scores """
        
        cos_sim = util.pytorch_cos_sim

        sent1_embedding = self.model.encode(text, convert_to_tensor=True)

        logging.info("compute cosine similarity between the embeddings")
        
        count=0
        
        for x in self.jobs:
            sent2_embedding = self.model.encode(x.description, convert_to_tensor=True)
            
            score = cos_sim(sent1_embedding, sent2_embedding)
            x.score=score
            count+=1
        
        self.jobs.sort(key=lambda x:x.score ,reverse=True)

    def test(self):
        """test performs the testing for the model, calling Evaluate to perform the evaluation."""
        logging.info("Testing beginning ")
        average_prec=0
        average_recall=0
        average_f1=0
        num=0
       
        for subset in self.test_jobs:
            logging.info("training a new subset is beginning")
            for x in subset:
                
                num+=1
                count=0
                self.findClosestMatch(x.description)
                
               
                target_industry=x.industry
                preds=[]
                
                for y in self.jobs:
                   
                    if count<30:
                        if y.industry==target_industry:
                            preds.append(1)
                        else:
                            preds.append(0)
                    else:
                        if y.industry==target_industry:
                            preds.append(0)
                        else:
                            preds.append(1)
                    count+=1
               
                e=Evaluate(preds)
                average_prec+=e.precision()
                
                average_recall+=e.recall()
               
                average_f1+=e.f1()
              
        logging.info("Testing ending ")
        print("Average precision is",average_prec/num,"average recall is",average_recall/num,"average f1 is",average_f1/num)






