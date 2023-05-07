
from transformers import  BertTokenizer#,# AdamW
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F
import torch 
from rake_nltk import Rake
import random
import matplotlib.pyplot as plt
import numpy as np
from parrot import Parrot
from sklearn import metrics
from sentence_transformers import SentenceTransformer,util
import scipy
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
    def __init__(self, token='bert-base-nli-mean-tokens', path = '../data/softwaredeveloper.csv'):
        self.path = path
        self.model = SentenceTransformer(token)
        self.jobs_df = pd.read_csv(path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.train_df, self.test_df = train_test_split(self.jobs_df, test_size=.2)
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
                    print("HEEERE")
                    loss = torch.mean(1 - cosine_similarity(embeddings, filtered_tensor))
                    loss.backward()
                    optimizer.step()
                else:
                # Compute loss using filtered tensor
                    print("HERE")
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


model = Job_Model( )
target_industry="Business Analyst"
#model = SentenceTransformer('bert-base-nli-mean-tokens')
#model.trainModel()
text="I am a recent college graduate with skills in Python, SQL, Communication, and writing. Experience leading teams, presenting, and software development "

master_list=[]
df = pd.read_csv("data/businessanalyst.csv")
model.df=df

model.industry="Business Analyst"
model.createJobList()
#model.trainModel(text)


model.findClosestMatch(text)

for x in model.jobs:
    master_list.append(x)

model2=Job_Model()
df = pd.read_csv("data/consultant.csv")
model2.df=df
model2.industry="Consultant"
model2.createJobList()
model2.findClosestMatch(text)

for x in model2.jobs:
    master_list.append(x)

model3=Job_Model()
df = pd.read_csv("data/dataengineer.csv")
model3.df=df
model3.industry="Data Engineer"
model3.createJobList()
model3.findClosestMatch(text)

for x in model3.jobs:
    master_list.append(x)

model4=Job_Model()
df = pd.read_csv("../data/datascience.csv")
model4.df=df
model4.industry="Data Science"
model4.createJobList()
model4.findClosestMatch(text)

for x in model4.jobs:
    master_list.append(x)

model5=Job_Model()
df = pd.read_csv("../data/financialnalyst.csv")
model5.df=df
model5.industry="Financial Analyst"
model5.createJobList()
model5.findClosestMatch(text)

for x in model5.jobs:
    master_list.append(x)

model6=Job_Model()
df = pd.read_csv("../data/healthcaremanagement.csv")
model6.df=df
model6.industry="Healthcare Management"
model6.createJobList()
model6.findClosestMatch(text)

for x in model6.jobs:
    master_list.append(x)

model7=Job_Model()
df = pd.read_csv("../data/ITroles.csv")
model7.df=df
model7.industry="IT Roles"
model7.createJobList()
model7.findClosestMatch(text)

for x in model7.jobs:
    master_list.append(x)

model8=Job_Model()
df = pd.read_csv("../data/marketing.csv")
model8.df=df
model8.industry="Marketing"
model8.createJobList()
model8.findClosestMatch(text)

for x in model8.jobs:
    master_list.append(x)

model9=Job_Model()
df = pd.read_csv("/data/productmanager.csv")
model9.df=df
model9.industry="Product Manager"
model9.createJobList()
model9.findClosestMatch(text)

for x in model9.jobs:
    master_list.append(x)

model10=Job_Model()
df = pd.read_csv("qualityassurance.csv")
model10.df=df
model10.industry="Quality Assurance"
model10.createJobList()
model10.findClosestMatch(text)


for x in model10.jobs:
    master_list.append(x)

model11=Job_Model()
df = pd.read_csv("../data/softwaredeveloper.csv")
model11.df=df
model11.industry="Software Developer"
model11.createJobList()
model11.findClosestMatch(text)

for x in model8.jobs:
    master_list.append(x)

master_list.sort(key=lambda x:x.score ,reverse=True)
for x in master_list:

    print(x.score,x.index,x.industry,x.description)
    input()
count=0
predictions=[]
for x in master_list:
    print(x.industry)
    if(count!=31):
        
        if x.industry==target_industry:
            
            predictions.append(1)
        else:
            predictions.append(0)
    else:
        if x.industry==target_industry:
            predictions.append(0)
        else:
            predictions.append(1)
    count+=1
print(predictions)
e=Evaluate(preds=predictions)
print(e.precision())
print(e.recall())

e.confusion_matrix()





