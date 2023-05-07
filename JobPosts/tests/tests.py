import sys
import os
import pandas as pd

path = os.path.abspath("../../data/")
path2 = os.path.abspath("../../JobPosts/")

sys.path.append(path)
sys.path.append(path2)
from eval.eval import Evaluate
from utils.model import Job, Job_Model


model=Job_Model()
df = pd.read_csv("../../data/businessanalyst.csv")
model.df=df

model.industry="Business Analyst"

model.createJobList()


def test_top_match():
    toFind=model.jobs[0].description
    model.findClosestMatch(model.jobs[0].description)
    assert(model.jobs[0].description==toFind)

def test_Precision():
    preds=[1 for x in range (40)]
    real=[1 for x in range(40)]
    e=Evaluate(preds)
    e.real=real
    assert(e.precision()==1)
   
def test_recall():
    preds=[1 for x in range (40)]
    real=[1 for x in range(40)]
    e=Evaluate(preds)
    e.real=real
    assert(e.recall()==1)
def test_f1():
    preds=[1 for x in range (40)]
    real=[1 for x in range(40)]
    e=Evaluate(preds)
    e.real=real
    assert(e.f1()==1)

    