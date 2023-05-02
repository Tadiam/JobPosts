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



model = Job_Model( )
target_industry="Financial Analyst"

#model.trainModel()
text="Five or more years related experience in a business intelligence, financial reporting, financial analyst, accounting analyst, public accounting, financial…"

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




df = pd.read_csv("../../data/datascience.csv")
model.df=df
model.industry="Data Science"
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

model.trainModel()

model.findClosestMatch(text)

for x in model.jobs:
    master_list.append(x)

master_list.sort(key=lambda x:x.score ,reverse=True)
count=0
for x in master_list:

    print(x.score,x.index,x.industry,x.description)
   
count=0
predictions=[]
for x in master_list:
    #print(x.industry)
    if(count!=31):
        
        if x.industry==target_industry:
            print(x.description,x.score)
          
            predictions.append(1)
        else:
            predictions.append(0)
    else:
        if x.industry==target_industry:
            print(x.description,x.score)
  
            predictions.append(0)
        else:
            predictions.append(1)
    count+=1

#print(predictions)
e=Evaluate(preds=predictions)
print(e.precision())
print(e.recall())

e.confusion_matrix()