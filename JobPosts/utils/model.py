
from transformers import  BertTokenizer#,# AdamW
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F
import torch 
from rake_nltk import Rake
import random
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sentence_transformers import SentenceTransformer,util

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import wordnet as wn


class Job:
    def __init__(self,description,index,industry):
        self.description=description
        self.index=index
        self.score=0
        self.paraphrase=""
        self.industry=industry
        
class Job_Model:
    def __init__(self,  token='bert-base-nli-mean-tokens', ):
        
        self.model = SentenceTransformer(token)
        df=""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.train_df, self.test_df = train_test_split(self.jobs_df, test_size=.2)
        self.jobs=[]
        self.industry=""
        self.train_jobs=[]
        self.test_jobs=[]
        self.length=0
    def __str__(self):
        return f'Job Model for data set with path {self.path}'
    def __repr__(self):
        return f'Job Model, path (\'{self.path}\''
    def createJobList(self):
        count=0
        for x in self.df["Short Description"]:
            new_job=Job(description=x,index=count,industry=self.industry)
            count+=1
            self.jobs.append(new_job)
        self.seperateData()
        self.length=len(self.jobs)
       
    def seperateData(self):
       
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
        # Fine-tune BERT model on dataset of resumes and job descriptions
        #train_texts = ['Responsible for conceiving and creating technical content in a variety of media such as technical articles, white papers, blogs, videos, and eBooks.', 'Creating technical content and other media. Adept at writing papers, blogs, and online books',"chef who loves to bake and play football","a teacher who hates technology andd is passionate about bringing literature back to paper","marketer with decades of experience and loves the idea of bringing companies into the modern world with technical skills", "an artist who loves to write on paper, not adept with technology"]
        length=0
        texts_against=[]
        length=0
        for s in self.train_jobs:
            for u in s:
                texts_against.append(u.description)
                length+=1
        for subset in self.train_jobs:
           print("H")
           for x in subset:
             
                new_array = []
                for i in self.train_jobs:
                    for j in i:
                        if j.description in subset:
                            new_array.append(1)
                        else:
                            new_array.append(0)
                train_texts = [x.description for l in range(length)]
                
               
             
                train_encodings = self.model.encode((train_texts))
                train_set_encoding=self.model.encode(texts_against)
                train_labels = new_array
                print(len(train_encodings),len(train_set_encoding),len(train_labels))
                train_dataset = TensorDataset(torch.tensor(train_encodings), torch.tensor(train_set_encoding), torch.tensor(train_labels))


                train_dataloader = DataLoader(train_dataset, batch_size=10)

                optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

                self.model.train()
                for num in range(3):
                
                    for batch in train_dataloader:
                       
                        optimizer.zero_grad()
                        train_embeddings,texts_against_embeddings, label = batch
                        train_embeddings.requires_grad = True
                        texts_against_embeddings.requires_grad = True
                        #label.requires_grad = True
                           
                        cos_sim = util.pytorch_cos_sim
                        cosine_similarities = cos_sim(train_embeddings, texts_against_embeddings)
                        loss = 1-torch.mean(cosine_similarities)
                        loss.backward()
                        optimizer.step()
                # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # self.model.to(device)
                # for epoch in range(3):

                #     for batch in train_dataloader:
                #         optimizer.zero_grad()

                #         input_ids, attention_mask, texts_against_input_ids, texts_against_attention_mask, labels = batch
                #         input_ids = input_ids.to(device)
                #         attention_mask = attention_mask.to(device)
                #         texts_against_input_ids = texts_against_input_ids.to(device)
                #         texts_against_attention_mask = texts_against_attention_mask.to(device)
                #         labels = labels.to(device)

                #         train_embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
                #         texts_against_embeddings = self.model(input_ids=texts_against_input_ids, attention_mask=texts_against_attention_mask)['last_hidden_state']


                #         cos_sim = util.pytorch_cos_sim
                #         cosine_similarities = cos_sim(train_embeddings, texts_against_embeddings)

                #         loss = 1 - torch.mean(cosine_similarities)
                #         loss.backward()
                #         optimizer.step()

    #RUNS ON TEST DATA
    #def findClosestMatch(resume_text,df,model):
    
    def findClosestMatch(self,text):
        #trainModel(model,df)
        
        # test_texts = []
        # for i in self.test_df['Short Description']:
        #     paraphrase(i)
        #this is paraphrasing the original description to search for this job 
        #vector1=self.model.encode(test_texts)
        
        # for x in self.jobs:
        #     new=x.paraphrase(x.description)
        #     x.paraphrase=new
        
        cos_sim = util.pytorch_cos_sim

        sent1_embedding = self.model.encode(text, convert_to_tensor=True)
        

      
       
       

        # compute cosine similarity between the embeddings
        
        
        count=0
        
        for x in self.jobs:
            vector2 = self.model.encode(str(x.description))
           
            sent2_embedding = self.model.encode(x.description, convert_to_tensor=True)
            
            score = cos_sim(sent1_embedding, sent2_embedding)
            x.score=score
            count+=1
        
        self.jobs.sort(key=lambda x:x.score ,reverse=True)
    def test(self):
        pass