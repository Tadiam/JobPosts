
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
    
    def trainModel(self,text):        
        # Fine-tune BERT model on dataset of resumes and job descriptions
        #train_texts = ['Responsible for conceiving and creating technical content in a variety of media such as technical articles, white papers, blogs, videos, and eBooks.', 'Creating technical content and other media. Adept at writing papers, blogs, and online books',"chef who loves to bake and play football","a teacher who hates technology andd is passionate about bringing literature back to paper","marketer with decades of experience and loves the idea of bringing companies into the modern world with technical skills", "an artist who loves to write on paper, not adept with technology"]
        
        
        train_texts = [x.description for x in self.jobs]
        
        train_encodings = self.model.encode(train_texts)
        train_labels = [1 for x in range (len(self.jobs))]
        train_dataset = TensorDataset(torch.tensor(train_encodings),
                                    torch.tensor(train_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=16)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.model.train()
        for epoch in range(3):
            for batch in train_dataloader:
                optimizer.zero_grad()
                text = batch[0]
                
                # Ensure that text is a tensor of strings
                if text.ndim == 0:
                    text = torch.tensor([str(text)])
                else:
                    text = text[torch.tensor([isinstance(t, str) for t in text])]
                    
                if len(text) == 0:
                    embeddings = None
                else:
                    embeddings = self.model.encode(text)
                
                
                # Filter out non-string elements from batch[0] tensor
                text_list = batch[0].tolist()

    # Filter out any non-string elements
                filtered_list = [t for t in text_list if isinstance(t, str)]

                # Convert filtered list back to a tensor
                filtered_tensor = torch.tensor(filtered_list)
                if len(filtered_tensor) > 0:
                  
                    loss = torch.mean(1 - cosine_similarity(embeddings, filtered_tensor))
                    loss.backward()
                    optimizer.step()
                else:
                # Compute loss using filtered tensor
              
                    loss = torch.mean(1 - cosine_similarity(embeddings, filtered_tensor))
                    loss.backward()
                    optimizer.step()
        

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