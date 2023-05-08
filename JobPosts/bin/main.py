""" main.py processes user input, comparing it to our model of job description data.
    
    main.py creates a new model based on the data files stored in the data folder. 
    It calls functions in the model module which train and test the model.
    It also parses input, taking in a string description of the user's qualifications and experiences,
    comparing it to the jobs stored in the model, and then printing out the
    jobs the user should apply to in ranked order based on similarity score.

"""
import sys
import os
import pandas as pd

path = os.path.abspath("../../data/")
path2 = os.path.abspath("../../JobPosts/")
#(path)
#print(path2)
sys.path.append(path)
sys.path.append(path2)
from eval.eval import Evaluate
from utils.model import Job, Job_Model
import argparse

parser = argparse.ArgumentParser(description="What args")
parser.add_argument('--sentence',type=str)
parser.add_argument('--n',type=int)
args=parser.parse_args()
model = Job_Model( )



master_list=[]
df = pd.read_csv("../../data/businessanalyst.csv")
model.df=df

model.industry="Business Analyst"

model.createJobList()





df = pd.read_csv("../../data/consultant.csv")
model.df=df
model.industry="Consultant"
model.createJobList()





df = pd.read_csv("../../data/dataengineer.csv")
model.df=df
model.industry="Data Engineer"
model.createJobList()







df = pd.read_csv("../../data/financialnalyst.csv")
model.df=df
model.industry="Financial Analyst"
model.createJobList()



df = pd.read_csv("../../data/healthcaremanagement.csv")
model.df=df
model.industry="Healthcare Management"
model.createJobList()

df = pd.read_csv("../../data/ITroles.csv")
model.df=df
model.industry="IT Roles"
model.createJobList()

df = pd.read_csv("../../data/marketing.csv")
model.df=df
model.industry="Marketing"
model.createJobList()

df = pd.read_csv("../../data/productmanager.csv")

model.df=df
model.industry="Product Manager"
model.createJobList()

df = pd.read_csv("../../data/qualityassurance.csv")
model.df=df
model.industry="Quality Assurance"
model.createJobList()


df = pd.read_csv("../../data/softwaredeveloper.csv")
model.df=df
model.industry="Software Developer"
model.createJobList()

df = pd.read_csv("../../data/datascience.csv")
model.df=df
model.industry="Data Science"
model.createJobList()
#model.trainModel()

model.test()

if(args.sentence=="default"):
    model.test()
else:
    (model.findClosestMatch(args.sentence))
    for x in range(0,args.n):
        print(model.jobs[x].industry,model.jobs[x].description)

