# JobPosts
Final Project for Computational Linguistics and Advanced Python


When applying for jobs in and around the tech industry, titles of postings are not always standard. There’s nuances to each position that differs from company to company. “Quality Engineer” or “Program Manager” at one place may not mean the same thing as at another. The qualifications, experiences, and job details may make them seem like completely different jobs, despite their shared title. This may negatively impact people’s job search experience, especially if they do not have a high level industry knowledge, despite being qualified. Our model will take a set of qualifications, salary expectations, and other job search criteria to recommend candidates with jobs that fits their search criteria and resume, not just one that has a similar title to what they’ve already applied to.

#Running Our Model

To run our model, there are two options: 

1. Run our train and test functions to see how our model works with our data. This will also run our evaluation feature and To do this, run our main.py file in this form:

python3 main.py default

2. Run the train function and then have the model find job matches for you. This will not run the evaluations but will run the training function and test function to find you matches. To do this, run main.py in this form:

python3 main.py "YOUR QUALIFIICATIOS AND EXPERIECES IN QUOTES"

NOTE: Depending on your machine, our model takes quite a long time to run all the way. In case you don't want to sit through iterations of using our model, we've included some data from our tests.


