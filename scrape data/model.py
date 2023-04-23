from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch.nn.functional as F
import torch 

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)



jobs_df = pd.read_csv('../Indeedatascientistdatascraped.csv')

resume_text = "SQL"
encoded_resume = tokenizer.encode_plus(resume_text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')

job_texts = jobs_df['Short Description'].tolist()
encoded_jobs = tokenizer.batch_encode_plus(job_texts, max_length=128, padding='max_length', truncation=True, return_tensors='pt')

with torch.no_grad():
    resume_embedding = model(**encoded_resume)[0]
    job_embeddings = model(**encoded_jobs)[0]

import torch.nn.functional as F

cos_similarities = F.cosine_similarity(resume_embedding, job_embeddings, dim=1)
best_match_index = cos_similarities.argmax().item()
best_match_job = jobs_df.iloc[best_match_index]
print(best_match_job)
print(best_match_index)
