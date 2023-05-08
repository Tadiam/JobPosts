# JobPosts Python Package
Authors: Adiam Tesfaselassie, Emily Smith, Zoë Moore

Final Project for Computational Linguistics and Advanced Python


When applying for jobs in and around the tech industry, titles of postings are not always standard. There’s nuances to each position that differ from company to company. “Quality Engineer” or “Program Manager” at one place may not mean the same thing as at another. The qualifications, experiences, and job details may make them seem like completely different jobs, despite their shared title. This may negatively impact people’s job search experience, especially if they do not have a high level industry knowledge, despite being qualified. 

Given recent tech layoffs, many high-skilled workers are looking for more suggestions of the jobs they are qualified for. We took scrapings of jobs in several roles: businessanalyst, consultant, dataengineer, datascience, financialanalyst, healthcaremanagement, ITroles, marketing, productmanager, qualityassurance, and software developer, to use as data for recommendations for our user. In our results, we found that many descriptions do match the role you would expect, but that a given input returns a significant amount of description outside a user input's intended job title description. As our system returns job postings ranked in terms of relevance, the user can scroll through a smaller, pointed amount of recommendations for listings that are similar to their previous job experiences, as well as ones that match their interests. Our system also allows users to easily modify descriptions, in case they want to look for new jobs that focus on certain skills more than others to get more suggestions.

We used an object-oriented approach to package our data: for example, our Job_Model class that stores the model's data and relevant functionality, and our Evaluate class in utils is used for evaluation with several functions corresponding to different metrics. Our model takes set of qualifications, salary expectations, and other job search criteria to recommend candidates with jobs that better their search criteria and resume, which they wouldn't find searching only for job titles that they think they're prepared for.

# Running Our Model

To run our model, there are two options: 

1. Run our train and test functions to see how our model works with our data. This will also run our evaluation feature and To do this, run our main.py file in this form:

python3 main.py --sentence "default"

2. Run the train function and then have the model find job matches for you. This will not run the evaluations but will run the training function and test function to find you matches. To do this, run main.py in this form:

python3 main.py --sentence "YOUR QUALIFICATIONS AND EXPERIENCES IN QUOTES" --n "Number of results you wish to see (don't put in quotes)

<img width="1146" alt="Screen Shot 2023-05-06 at 8 06 49 PM" src="https://user-images.githubusercontent.com/91433035/236651231-f0261b11-2e64-438b-95b6-715f92121878.png">

NOTE: Depending on your machine, our model takes quite a long time to run all the way. In case you don't want to sit through iterations of using our model, we've included some data from our tests.

Running an exact job description to find the most relevant:

<img width="676" alt="Screen Shot 2023-05-06 at 7 36 06 PM" src="https://user-images.githubusercontent.com/91433035/236650492-034b84b9-c22b-4095-9343-329341a2258e.png">


To run our Pytests navigate to the test directory and run

pytest tests.py


# Limitations/Ideas for Future Projects

One limitation of our project was the amount of data found through our job scraping process, and the amount that our individual laptops' CPU power could handle. Future projects could collect more job scrapings by automating a system scraping websites every day and building a larger database as a result. GPU services could run our model much faster than our computer systems could. This model could also be more useful for people across different job sectors. It could be fine-tuned on a variety of other job datasets.

Another limitation we faced was in switching our local interface to a website host that could make our project widely available. Our website ran and trained our model, but we faced time limitations that prevented us from figuring out the SQL configuration to store our model as a database and do instantaneous comparisons. The development we've done thus far requires the model to run each time as a result. Unfortunately, our tool isn't available through a website interface like we were developing and like we had hoped for. 

# Happy Job Hunting!




