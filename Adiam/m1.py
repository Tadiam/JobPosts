
from transformers import  BertTokenizer#,# AdamW
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F
import torch 
from rake_nltk import Rake
import random
import numpy as np

from sentence_transformers import SentenceTransformer
import scipy
from sklearn.metrics.pairwise import cosine_similarity


def trainModel(model):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Fine-tune BERT model on dataset of resumes and job descriptions
    train_texts = ['Responsible for conceiving and creating technical content in a variety of media such as technical articles, white papers, blogs, videos, and eBooks.', 'Creating technical content and other media. Adept at writing papers, blogs, and online books',"chef who loves to bake and play football","a teacher who hates technology andd is passionate about bringing literature back to paper","marketer with decades of experience and loves the idea of bringing companies into the modern world with technical skills", "an artist who loves to write on paper, not adept with technology"]
    train_encodings = model.encode(train_texts)
    train_labels = [1, 1, 0,0,1,0]
    
    train_dataset = TensorDataset(torch.tensor(train_encodings),
                              torch.tensor(train_labels))
    train_dataloader = DataLoader(train_dataset, batch_size=16)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    model.train()
    for epoch in range(7):
        for batch in train_dataloader:
            optimizer.zero_grad()
            text = batch[0]
            if len(text.shape) == 0:  # Check if tensor is empty or has shape (1,)
                text = text.unsqueeze(0) # Add extra dimension to create shape (1, 1)
            text = [(str(t)) for t in text]
            embeddings = model.encode(text)
            loss = 1 - cosine_similarity(embeddings, batch[0]).mean()
            loss_tensor = torch.tensor(loss, requires_grad=True)
            loss_tensor.backward()
           
            optimizer.step()

#RUNS ON TEST DATA
def findClosestMatch(resume_text,df,model):
   
    trainModel(model)
    vector1=model.encode(resume_text)
    count=0
    s=0
    for x in df['Short Description']:
        vector2 = model.encode(x)
        score = 1-(scipy.spatial.distance.cosine(vector1, vector2))
        if score>s:
            s=score
            index=count
        count+=1
    
    print(df.iloc[index]['Position Name'])
    return(score)

model = SentenceTransformer('bert-base-nli-mean-tokens')
path = '../data/'
jobs = glob.glob(path + "/*.csv")
df_list = (pd.read_csv(file) for file in jobs)
jobs_df = pd.concat(df_list)
jobs_df

#jobs_df = pd.read_csv('../data/marketing.csv')
resume_text= "Knowledge of high-level design thinking and network development.Technical skills in information science, code, development and design."
#trainModel(model)
print(findClosestMatch(resume_text,jobs_df,model))