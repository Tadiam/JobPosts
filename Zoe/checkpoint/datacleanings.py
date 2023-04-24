import pandas as pd
import matplotlib as plt 
import sklearn as sk
import glob
import os
import sys

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
import pandas as pd
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

def main():
    #path = os.path.abspath("../data")
    #sys.path.append(path)

    path = '../data'

    jobs = glob.glob(path + "/*.csv")
    df_list = (pd.read_csv(file) for file in jobs)
    jobs_df = pd.concat(df_list)
    #print(jobs_df.to_string())
    #df_jobs = pd.concat([pd.read_csv(f).set_index('A') for f in glob.glob('/Users/adiamtesfaselassie/Documents/Spring2023/ANLY521/JobPosts/Adiam/data')])

    discription_df = pd.DataFrame(zip(jobs_df['Short Description'], jobs_df['Job']), columns=['Description', 'Job'])
    discription_df.Job.unique()
    #print(discription_df.to_string())

    lsa = TruncatedSVD(2) # issue point 
    array,doc = return_topics(discription_df['Description'],20, 10, NMF, TfidfVectorizer)
    #print(doc.tolist())

    # look at topics and see what makes sense

    # recommend roles that are best for you based on supervised learnings

    # these are the jobs descriptions that are best for you based on your resume

    # these are the best words for your resume based on which job you want to go into

    Topic_DF = pd.DataFrame(doc)
    Topic_DF.columns = ['Topic ' + str(i+1) for i in range(len(Topic_DF.columns)) ]
    print(Topic_DF.to_string())

def tokenize_stem(series):
    
    tokenizer = TreebankWordTokenizer() 
    stemmer = PorterStemmer()
    series = series.apply(lambda x: tokenizer.tokenize(x))
    series = series.apply(lambda x: [stemmer.stem(w) for w in x])
    series = series.apply(lambda x: ' '.join(x))
    return series

def display_topics(model, feature_names, no_top_words, topic_names=None):
    for i, topic in enumerate(model.components_):
        if not topic_names or not topic_names[i]:
            print("\nTopic ", i)
        else:
            print("\nTopic: '",topic_names[i],"'")
        print(", ".join([feature_names[k]
                        for k in topic.argsort()[:-no_top_words - 1:-1]]))
    return [topic for topic in model.components_]
    return model.components_

def return_topics(series, num_topics, no_top_words, model, vectorizer):
    #turn job into series
    #df = jobs_df[jobs_df['Job']==job]
    
    
    #clean series
    
    series = tokenize_stem(series)
    #transform series into corpus
    ex_label = [e[:30]+"..." for e in series]
    #set vectorizer ngrams = (2,2)
    vec = vectorizer(stop_words = 'english')
    
    doc_word = vec.fit_transform(series)
    def_model = model(num_topics)
    doc_topic = def_model.fit_transform(doc_word)
    
    display_topics(def_model, vec.get_feature_names_out(), no_top_words)
    return def_model.components_, doc_topic#, topics


if __name__ == "__main__":
    main()
