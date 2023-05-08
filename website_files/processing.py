from model import Job_Model
from handle_database import get_our_model
from sentence_transformers import util

def process(query):

    model_copy = get_our_model()

    cos_sim = util.pytorch_cos_sim

    sent1_embedding = model_copy.model.encode(query, convert_to_tensor=True)

    count=0

    for x in model_copy.jobs:
        sent2_embedding = model_copy.model.encode(x.description, convert_to_tensor=True)

        score = cos_sim(sent1_embedding, sent2_embedding)
        x.score=score
        count+=1
    model_copy.jobs.sort(key=lambda x:x.score ,reverse=True)
   

    return "Here are the recommended jobs from our databases, ranked for your description: {r}".format(r=model_copy.jobs)
