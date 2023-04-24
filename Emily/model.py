from transformers import AutoTokenizer, AutoModelForSequenceClassification,BertForSequenceClassification,BertTokenizer
import pandas as pd
import torch.nn.functional as F
import torch 
from rake_nltk import Rake
import random
import numpy as np

from sentence_transformers import SentenceTransformer
import scipy





def findClosestMatch(resume_text,df):
   
    model = SentenceTransformer('bert-base-nli-mean-tokens')
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

jobs_df = pd.read_csv('../data/marketing.csv')
resume_text= "Knowledge of high-level design thinking and network development.Technical skills in information science, code, development and design."
print(findClosestMatch(resume_text,jobs_df))

