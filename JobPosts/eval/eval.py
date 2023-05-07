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
        for x in range (0,30):
            
            if (self.predictions[x]==1 and self.real[x]==1):
                num=num+1
        false_negative=31-num
        
        return(num/(num+false_negative))
            

    def precision(self):
       # print(self.predictions)
        #print(self.recall)
        num=0
        for x in range (0,30):
           
            if (self.predictions[x]==1 and self.real[x]==1):
                #print("HERE")
                num=num+1
        #print(num)
        false_positive=31-num
        return(num/(num+false_positive))
    def f1(predictions,real):
        pass
    def confusion_matrix(self):
        confusion_matrix = metrics.confusion_matrix(self.real, self.predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        cm_display.plot()
        plt.show()