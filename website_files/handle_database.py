import pandas as pd

from model import Job_Model


our_model = Job_Model( )



master_list=[]
df = pd.read_csv("./data/businessanalyst.csv")
our_model.df=df

our_model.industry="Business Analyst"

our_model.createJobList()

df = pd.read_csv("./data/consultant.csv")
our_model.df=df
our_model.industry="Consultant"
our_model.createJobList()

df = pd.read_csv("./data/dataengineer.csv")
our_model.df=df
our_model.industry="Data Engineer"
our_model.createJobList()

df = pd.read_csv("./data/financialnalyst.csv")
our_model.df=df
our_model.industry="Financial Analyst"
our_model.createJobList()

df = pd.read_csv("./data/healthcaremanagement.csv")
our_model.df=df
our_model.industry="Healthcare Management"
our_model.createJobList()

df = pd.read_csv("./data/ITroles.csv")
our_model.df=df
our_model.industry="IT Roles"
our_model.createJobList()

df = pd.read_csv("./data/marketing.csv")
our_model.df=df
our_model.industry="Marketing"
our_model.createJobList()

df = pd.read_csv("./data/productmanager.csv")

our_model.df=df
our_model.industry="Product Manager"
our_model.createJobList()

df = pd.read_csv("./data/qualityassurance.csv")
our_model.df=df
our_model.industry="Quality Assurance"
our_model.createJobList()


df = pd.read_csv("./data/softwaredeveloper.csv")
our_model.df=df
our_model.industry="Software Developer"
our_model.createJobList()

df = pd.read_csv("./data/datascience.csv")
our_model.df=df
our_model.industry="Data Science"
our_model.createJobList()
our_model.trainModel()

our_model.test()

def get_our_model():
    return our_model

#if(args.sentence=="default"):
#    our_model.test()
#else:
#    (our_model.findClosestMatch(args.sentence))
#    for x in range(0,args.n):
#        print(our_model.jobs[x].industry,our_model.jobs[x].description)

