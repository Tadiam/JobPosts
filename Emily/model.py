
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
    def __init__(self, token='bert-base-nli-mean-tokens', path = '../data/seek_australia.csv'):
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
        print("going in looop")
        for x in self.train_df["job_description"] :
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
        for i in self.test_df['job_description']:
            test_texts.append(parrot.augment(input_phrase=i))
        #this is paraphrasing the original description to search for this job 
        vector1=self.model.encode(test_texts)
        count=0
        s=0
        for x in self.test_df['job_description']:
            vector2 = model.encode(str(x))
            score = 1-(scipy.spatial.distance.cosine(vector1, vector2))
            if score>s:
                s=score
                index=count
            count+=1
        
        print(self.jobs_df.iloc[index]['job_description'])
        return(score)

model = Job_Model(SentenceTransformer('bert-base-nli-mean-tokens'), '../data/seek_australia.csv')
#model = SentenceTransformer('bert-base-nli-mean-tokens')
model.trainModel()
print(model.findClosestMatch("input"))

<<<<<<< HEAD
    resume_encodings = model.encode(train_texts)
    print("train done")
    job_encodings=model.encode(descriptions)
    print("job done")
    #train_labels=[1,1,0,0,1]
    print("boutta loop")
    train_labels = [1 for x in range(len(df))]
    print("HIIIIII")
    train_dataset = TensorDataset(torch.tensor(resume_encodings),
                        torch.tensor(job_encodings),
                        torch.tensor(train_labels))
    train_dataloader = DataLoader(train_dataset, batch_size=16)
=======
#trainModel(model, jobs_df)
>>>>>>> 80c9bbad5c056c5add4eb6b7b21acdaeb1b5a190


<<<<<<< HEAD
#RUNS ON TEST DATA
def findClosestMatch(resume_text,df,model):
   
    trainModel(model,df)
    print("hereeeee")
    vector1=model.encode(resume_text)
    count=0
    s=0
    print("going here")
    
    for x in df['job_description']:
       
        vector2 = model.encode(str(x))
        score = 1-(scipy.spatial.distance.cosine(vector1, vector2))
        if score>s:
            s=score
            index=count
        count+=1
        if count ==10000:
            break
    
    print(df.iloc[index]['job_description'])
    return(score)

model = SentenceTransformer('bert-base-nli-mean-tokens')

jobs_df = pd.read_csv('../data/seek_australia.csv')
resume_text= "Ensure the stability, confidentiality, integrity & availability of services Grow & develop capabilities & efficiencies of the services across the CSIRO enterprise Ongoing professional & development opportunities within a friendly, supportive team Our Information Management and Technology (IMT) team are looking for an IT Directory Services & Email Support Analyst who has experience supporting the following technologies at an Enterprise scale: Active Directory, DHCP/DNS on Windows and Linux, Exchange.Â  You will be required to work with immediate team members as well as geographically and technically distributed teams across the architecture model to grow and develop capabilities and efficiencies of the services across the CSIRO enterprise.Â  You will have responsibility for: the completion of complex technical problems, undertaking development, implementation or standardisation of procedures and techniques, and input to solutions design. To be eligible for this position you will hold a Degree in information technology and/or have equivalent work experience. You will have demonstrated experience in supporting the following technologies at an enterprise scale: Active Directory; DHCP/DNS on Windows and Linux; Exchange.Â  Experience contributing to the implementation and administration of enterprise IT solutions in a converged IT environment is also required. This is a security assessed position. To be eligible for this position you will currently hold, or will have the ability to obtain, an Australian Government security clearance level of Negative Vetting 1 (SECRET). Location:Â Â Â  Clayton, VIC; Yarralumla, ACT or North Ryde,NSW Salary:Â Â Â Â Â Â Â Â AU $61K to AU $78K plus up to 15.4% superannuation Tenure:Â Â Â Â Â Â  Indefinite Ref No.:Â Â Â Â Â Â 56625 Before applying please view the full position details and selection criteria here:Â Position Details About CSIRO We imagine. We collaborate. "
#trainModel(model)
print("here")
print(findClosestMatch(resume_text,jobs_df,model))
=======
#resume_text= "Ensure the stability, confidentiality, integrity & availability of services Grow & develop capabilities & efficiencies of the services across the CSIRO enterprise Ongoing professional & development opportunities within a friendly, supportive team Our Information Management and Technology (IMT) team are looking for an IT Directory Services & Email Support Analyst who has experience supporting the following technologies at an Enterprise scale: Active Directory, DHCP/DNS on Windows and Linux, Exchange.Â  You will be required to work with immediate team members as well as geographically and technically distributed teams across the architecture model to grow and develop capabilities and efficiencies of the services across the CSIRO enterprise.Â  You will have responsibility for: the completion of complex technical problems, undertaking development, implementation or standardisation of procedures and techniques, and input to solutions design. To be eligible for this position you will hold a Degree in information technology and/or have equivalent work experience. You will have demonstrated experience in supporting the following technologies at an enterprise scale: Active Directory; DHCP/DNS on Windows and Linux; Exchange.Â  Experience contributing to the implementation and administration of enterprise IT solutions in a converged IT environment is also required. This is a security assessed position. To be eligible for this position you will currently hold, or will have the ability to obtain, an Australian Government security clearance level of Negative Vetting 1 (SECRET). Location:Â Â Â  Clayton, VIC; Yarralumla, ACT or North Ryde,NSW Salary:Â Â Â Â Â Â Â Â AU $61K to AU $78K plus up to 15.4% superannuation Tenure:Â Â Â Â Â Â  Indefinite Ref No.:Â Â Â Â Â Â 56625 Before applying please view the full position details and selection criteria here:Â Position Details About CSIRO We imagine. We collaborate. "
#trainModel(model)

>>>>>>> 80c9bbad5c056c5add4eb6b7b21acdaeb1b5a190

