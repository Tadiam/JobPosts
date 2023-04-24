from transformers import AutoTokenizer, AutoModelForSequenceClassification,BertForSequenceClassification,BertTokenizer
import pandas as pd
import torch.nn.functional as F
import torch 
from rake_nltk import Rake
import random
import numpy as np
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# def tokenize_text(text):
#     # Tokenize the text using the BERT tokenizer
#     tokenized_text = tokenizer.encode_plus(
#         text,                      # Input text to encode
#         add_special_tokens=True,   # Add [CLS] and [SEP] tokens
#         max_length=512,           # Truncate the text to the maximum length
#         pad_to_max_length=True,   # Pad the text to the maximum length
#         return_attention_mask=True,  # Return attention mask
#         return_tensors='pt'       # Return PyTorch tensors
#     )
#     return tokenized_text

# # Define a function to match the skills to the job descriptions
# def match_skills_to_job_descriptions(skills, job_descriptions):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
    
#     # tokenize skills
#     skills_tokens = tokenize_text(skills)
#     skills_tokens = tokenizer.encode_plus(skills, add_special_tokens=True, return_attention_mask=True,
#                                            return_tensors='pt')
#     # loop through job descriptions and get similarity scores
#     scores = []
#     for desc in job_descriptions:
#         desc_tokens = tokenizer.encode_plus(desc, add_special_tokens=True, return_attention_mask=True,
#                                              return_tensors='pt')
        
#         # get the model's predictions
#         input_ids = desc_tokens['input_ids']
#         attention_mask = desc_tokens['attention_mask']
#         with torch.no_grad():
#             outputs = model(input_ids, attention_mask=attention_mask)
#         logits = outputs.logits  # extract the logits tensor
#         score = torch.sigmoid(logits)  # apply the sigmoid function to get the similarity score
#         scores.append(score.item())  # append the score to the list of scores
        
#     # create a pandas dataframe with the job descriptions and their corresponding scores
#     df = pd.DataFrame({'job_description': job_descriptions, 'similarity_score': scores})
    
#     return df

# r = Rake()
# resume_text="We are searching for an entry level data scientist with python experience who learns quickly and thrives in a fast-paced, innovative environment"
# r.extract_keywords_from_text(resume_text)
# l=' '.join(r.get_ranked_phrases())


# jobs_df = pd.read_csv('../Indeedatascientistdatascraped.csv')



# #encoded_resume = tokenizer.encode_plus(l, max_length=128, padding='max_length', truncation=True, return_tensors='pt')

# job_texts = jobs_df['Short Description'].tolist()
# jobs=[]
# for x in job_texts:
#     tester=Rake()
#     tester.extract_keywords_from_text(x)
#     l=' '.join(tester.get_ranked_phrases())
#     jobs.append(l)
    
# # encoded_jobs = tokenizer.batch_encode_plus(jobs, max_length=128, padding='max_length', truncation=True, return_tensors='pt')

# # with torch.no_grad():
# #     resume_embedding = model(**encoded_resume)[0]
# #     job_embeddings = model(**encoded_jobs)[0]

# # import torch.nn.functional as F

# # cos_similarities = F.cosine_similarity(resume_embedding, job_embeddings, dim=1)
# # best_match_index = cos_similarities.argmax().item()
# # best_match_job = jobs_df.iloc[best_match_index]
# # print(best_match_job)
# # print(best_match_index)


# #skills = "We are searching for an entry level data scientist with python experience who learns quickly and thrives in a fast-paced, innovative environment"

# similarity_df = match_skills_to_job_descriptions(resume_text, jobs)
# print(similarity_df)
# min=0
# descrip=""
# count=0
# for x in similarity_df['similarity_score']:
#     if x>min:
#         min=x
#         descrip=similarity_df.iloc[count]
#     count+=1
# print(descrip,min)


#MODEL 2
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# def rank_job_descriptions(skills, job_descriptions):
#     skill_tokens = tokenizer.encode(skills, add_special_tokens=False)
#     skill_ids = torch.tensor([skill_tokens])

#     job_scores = []
#     for job in job_descriptions:
#         job_tokens = tokenizer.encode(job, add_special_tokens=False)
#         job_ids = torch.tensor([job_tokens])
#         with torch.no_grad():
#             outputs = model(job_ids, token_type_ids=None, attention_mask=None)
#             logits = outputs[0]
#             score = torch.sigmoid(logits).tolist()[0][1] # use the probability of the positive label as the score
#             job_scores.append(score)
    
#     # sort job descriptions by score in descending order
#     ranked_jobs = [job for _, job in sorted(zip(job_scores, job_descriptions), reverse=True)]
    
#     return ranked_jobs

# r = Rake()
# resume_text="bioengineering"
# r.extract_keywords_from_text(resume_text)
# l=' '.join(r.get_ranked_phrases())


# jobs_df = pd.read_csv('../Indeedatascientistdatascraped.csv')
# job_texts = jobs_df['Short Description'].tolist()
# jobs=[]
# for x in job_texts:
#     tester=Rake()
#     tester.extract_keywords_from_text(x)
#     l=' '.join(tester.get_ranked_phrases())
#     jobs.append(l)
# ranked_jobs=rank_job_descriptions(l, jobs)
# print(ranked_jobs[1])


from sentence_transformers import SentenceTransformer
import scipy

# Load a pre-trained BERT-based model


# Define two example blocks of text with different numbers of sentences
# text1 = "We are searching for an entry level data scientist with python experience who learns quickly and thrives in a fast-paced, innovative environment"
# text2 = "Data scientist experienced in creating data solutions; knowledgeable in data mining and machine learningI; Skilled SQL developer. Quick learner. Good team player and self-organized. Known by peers for strong critical thinking and problem-solving abilities."





def findClosestMatch(resume_text,df):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    vector1 = model.encode(resume_text)
   
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

jobs_df = pd.read_csv('../Indeedatascientistdatascraped.csv')
resume_text= "Knowledge of high-level design thinking and network development.Technical skills in information science, code, development and design."
print(findClosestMatch(resume_text,jobs_df))

