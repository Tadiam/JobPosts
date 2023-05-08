"""tests.py tests crucial aspects of our project: closestmatch and evaluation metrics, on the bussinessanalyst data"""
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
    """calls functionality implemented by model's Job_Model class to find the closest match.
        Then, it asserts that this is the best match.
    """
    toFind=model.jobs[0].description
    model.findClosestMatch(model.jobs[0].description)
    assert(model.jobs[0].description==toFind)

    toFind=model.jobs[1].description
    model.findClosestMatch(model.jobs[1].description)
    assert(model.jobs[1].description==toFind)

def test_Precision():
    """Calls Evaluate's precision and then asserts that it is true to test that method"""
    preds=[1 for x in range (40)]
    real=[1 for x in range(40)]
    e=Evaluate(preds)
    e.real=real
    assert(e.precision()==1)
   
def test_recall():
    """Calls Evaluate's recall and then asserts that it is true to test that method"""

    preds=[1 for x in range (40)]
    real=[1 for x in range(40)]
    e=Evaluate(preds)
    e.real=real
    assert(e.recall()==1)

def test_f1():
    """Calls Evaluate's f1 and then asserts that it is true to test that method"""
    preds=[1 for x in range (40)]
    real=[1 for x in range(40)]
    e=Evaluate(preds)
    e.real=real
    assert(e.f1()==1)

    