from transformers import BertTokenizer#,# AdamW
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer,util

import torch

import logging

from sklearn.metrics.pairwise import cosine_similarity

import os
import sys


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
        logging.basicConfig(filename='train.log', encoding='utf-8', level=logging.INFO)
        length=0
        texts_against=[]
        length=0
        for s in self.train_jobs:
            for u in s:
                texts_against.append(u.description)
                length+=1
        for subset in self.train_jobs:
          # logging.info("New job position running, this might take a while.:Q")
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



    def test(self):
        #logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)
        logging.info("Testing beginning ")
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








