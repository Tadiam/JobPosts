
from transformers import  BertTokenizer#,# AdamW
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F
import torch 
from rake_nltk import Rake
import random
import numpy as np
from parrot import Parrot

from sentence_transformers import SentenceTransformer
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


class Job_Model:
    def __init__(self, token='bert-base-nli-mean-tokens', path = '../data/softwaredeveloper.csv'):
        self.path = path
        self.model = SentenceTransformer(token)
        self.jobs_df = pd.read_csv(path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.train_df, self.test_df = train_test_split(self.jobs_df, test_size=.2)
        
    def __str__(self):
        return f'Job Model for data set with path {self.path}'
    def __repr__(self):
        return f'Job Model, path (\'{self.path}\''
    def trainModel(self):        
        # Fine-tune BERT model on dataset of resumes and job descriptions
        #train_texts = ['Responsible for conceiving and creating technical content in a variety of media such as technical articles, white papers, blogs, videos, and eBooks.', 'Creating technical content and other media. Adept at writing papers, blogs, and online books',"chef who loves to bake and play football","a teacher who hates technology andd is passionate about bringing literature back to paper","marketer with decades of experience and loves the idea of bringing companies into the modern world with technical skills", "an artist who loves to write on paper, not adept with technology"]
        
        train_texts=[]
        descriptions=[]
        count=0
       
        for x in self.train_df["Short Description"] :
            train_texts.append(x)
            descriptions.append(x)
            count+=1
            #if count==8000: #we've split by 80/20 now
                #break
        print("out of loop")

        resume_encodings = self.model.encode(train_texts)
        job_encodings=self.model.encode(descriptions)
        #train_labels=[1,1,0,0,1]
        train_labels = [1 for x in range(len(self.train_df))]
        
        train_dataset = TensorDataset(torch.tensor(resume_encodings),
                            torch.tensor(job_encodings),
                            torch.tensor(train_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=16)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        self.model.train()
        for epoch in range(7):
            for batch in train_dataloader:
                optimizer.zero_grad()
                resume_embedding, job_embedding, label = batch
                cosine_similarities = cosine_similarity(resume_embedding, job_embedding)
                loss = torch.mean((cosine_similarities - label)**2)
                loss.backward()
                optimizer.step()
        

    #RUNS ON TEST DATA
    #def findClosestMatch(resume_text,df,model):
    def findClosestMatch(self):
        #trainModel(model,df)
        parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)
        test_texts = []
        for i in self.test_df['Short Description']:
            test_texts.append(parrot.augment(input_phrase=i))
        #this is paraphrasing the original description to search for this job 
        vector1=self.model.encode(test_texts)
        count=0
        s=0
        for x in self.test_df['Short Description']:
            vector2 = model.encode(str(x))
            score = 1-(scipy.spatial.distance.cosine(vector1, vector2))
            if score>s:
                s=score
                index=count
            count+=1
        
        print(self.jobs_df.iloc[index]['Short Description'])
        return(score)

model = Job_Model( )
#model = SentenceTransformer('bert-base-nli-mean-tokens')
model.trainModel()
print(model.findClosestMatch("input"))

class Evaluate():
    def precision(predictions,real):
        pass
    def recall(predictions,real):
        pass
    def f1(predictions,real):
        pass