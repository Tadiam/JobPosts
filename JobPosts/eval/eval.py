from sklearn import metrics
import matplotlib.pyplot as plt
class Evaluate():
    def __init__(self,preds,real=[1 for x in range (0,31)]):
        self.real=real
        for x in range(31,len(preds)):
            self.real.append(0)
        self.predictions=preds
    def recall(self):
        num=0
        false_negative=0
        for x in range (0,len(self.predictions)):
            
            if (self.predictions[x]==1 and self.real[x]==1):
                num=num+1
            if(self.predictions[x]==1 and self.real[x]==0):
                false_negative+=1
        
        return(num/(num+false_negative))
            

    def precision(self):
        num=0
        false_pos=0
        for x in range (0,len(self.predictions)):
            
            if (self.predictions[x]==1 and self.real[x]==1):
                num=num+1
            if(self.predictions[x]==0 and self.real[x]==1):
                false_pos+=1
        
        return(num/(num+false_pos))
    def f1(self):
        top=2*self.precision()*self.recall()
        bottom=self.precision()+self.recall()
        return top/bottom
    def confusion_matrix(self):
        confusion_matrix = metrics.confusion_matrix(self.real, self.predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        cm_display.plot()
        plt.show()