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
parser.add_argument("mode")
args=parser.parse_args()
model = Job_Model( )


#model.trainModel()


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


if(args.mode=="default"):
    model.test()
else:
    model.findClosestMatch(args.mode)
