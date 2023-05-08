"""eval.py contains the class Evaluate"""
from sklearn import metrics
import matplotlib.pyplot as plt
class Evaluate():
    """ The Evaluate class calculates recall, precision, f1, and confusion matrix for a given model."""
    def __init__(self,preds,real=[1 for x in range (0,31)]):
        """__init__ for Evaluate, storing the real scores and predictions."""
        self.real=real
        for x in range(31,len(preds)):
            self.real.append(0)
        self.predictions=preds

    def __str__(self):
        """__str__ for Evaluate class"""
        return f'Evaluation performance for job postings model {self.path}'

    def __repr__(self):
        """__repr__ for Evaluate class"""
        return f'Evaluate, path (\'{self.path}\''

    def recall(self):
        """'recall' returns the recall for a given model"""
        num=0
        false_negative=0
        for x in range (0,len(self.predictions)):
            
            if (self.predictions[x]==1 and self.real[x]==1):
                num=num+1
            if(self.predictions[x]==1 and self.real[x]==0):
                false_negative+=1
        
        return(num/(num+false_negative))   

    def precision(self):
        """'precision' computes the precision for a given model"""
        num=0
        false_pos=0
        for x in range (0,len(self.predictions)):
            
            if (self.predictions[x]==1 and self.real[x]==1):
                num=num+1
            if(self.predictions[x]==0 and self.real[x]==1):
                false_pos+=1
        
        return(num/(num+false_pos))

    def f1(self):
        """'f1 returns f1 scores for a given model"""
        top=2*self.precision()*self.recall()
        bottom=self.precision()+self.recall()
        return top/bottom

    def confusion_matrix(self):
        """'confusion_matrix displays a confusion_matrix for a given model"""
        confusion_matrix = metrics.confusion_matrix(self.real, self.predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        cm_display.plot()
        plt.show()