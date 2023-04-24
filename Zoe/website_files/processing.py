def process(query):
    recommendation = "NLP Data Analyst"
    #we will insert finished python module and do all the processing here
    #recommendation will return full information for the best fit based on our NLP model
    #flask_app.py runs the html, returning info
    #process will call functions needed to 

    return "Your recommended job is: {r}".format(r=recommendation)
    #to check on query:
    #''' Your query was {qu}\n
     #       Your recommended job is: {rec}
     #       '''.format(qu=query, rec = recommendation)