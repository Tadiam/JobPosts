import pandas as pd
from model import Job_Model, Evaluate

def process(query):
    model = Job_Model( )

    #model = SentenceTransformer('bert-base-nli-mean-tokens')
    #model.trainModel()
    text="Perform business systems analysis and support of company-wide IT business applications and ensure the successful performance of financial systems by performing"

    master_list=[]
    df = pd.read_csv("/data/businessanalyst.csv")
    model.df=df
    target_industry = "Business Analyst"
    model.industry="Business Analyst"
    model.createJobList()
    #model.trainModel(text)


    model.findClosestMatch(text)

    for x in model.jobs:
        master_list.append(x)

    model2=Job_Model()
    df = pd.read_csv("/data/consultant.csv")
    model2.df=df
    model2.industry="Consultant"
    model2.createJobList()
    model2.findClosestMatch(text)

    for x in model2.jobs:
        master_list.append(x)

    model3=Job_Model()
    df = pd.read_csv("/data/dataengineer.csv")
    model3.df=df
    model3.industry="Data Engineer"
    model3.createJobList()
    model3.findClosestMatch(text)

    for x in model3.jobs:
        master_list.append(x)

    model4=Job_Model()
    df = pd.read_csv("/data/datascience.csv")
    model4.df=df
    model4.industry="Data Science"
    model4.createJobList()
    model4.findClosestMatch(text)

    for x in model4.jobs:
        master_list.append(x)

    model5=Job_Model()
    df = pd.read_csv("/data/financialnalyst.csv")
    model5.df=df
    model5.industry="Financial Analyst"
    model5.createJobList()
    model5.findClosestMatch(text)

    for x in model5.jobs:
        master_list.append(x)

    model6=Job_Model()
    df = pd.read_csv("/data/healthcaremanagement.csv")
    model6.df=df
    model6.industry="Healthcare Management"
    model6.createJobList()
    model6.findClosestMatch(text)

    for x in model6.jobs:
        master_list.append(x)

    master_list.sort(key=lambda x:x.score ,reverse=True)
    for x in master_list:
        print(x.score,x.index,x.industry)

    count=0
    predictions=[]
    for x in master_list:
        print(x.industry)
        if(count==31):
            break
        else:
            if x.industry==target_industry:
                print("HEREEEEEE")
                predictions.append(1)
            else:
                predictions.append(0)
        count+=1
    #print(predictions)
    e=Evaluate(preds=predictions)
    #print(e.precision())
    #print(e.recall())






    #recommendation = "NLP Data Analyst"
    #we will insert finished python module and do all the processing here
    #recommendation will return full information for the best fit based on our NLP model

    return "Here are the recommended jobs from our databases, ranked for your description: {r}".format(r=master_list)
    #to check on query:
    #''' Your query was {qu}\n
     #       Your recommended job is: {rec}
     #       '''.format(qu=query, rec = recommendation)