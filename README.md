# JobPosts
Final Project for Computational Linguistics and Advanced Python


When applying for jobs in and around the tech industry, titles of postings are not always standard. There’s nuances to each position that differs from company to company. “Quality Engineer” or “Program Manager” at one place may not mean the same thing as at another. The qualifications, experiences, and job details may make them seem like completely different jobs, despite their shared title. This may negatively impact people’s job search experience, especially if they do not have a high level industry knowledge, despite being qualified. Our model will take a set of qualifications, salary expectations, and other job search criteria to recommend candidates with jobs that fits their search criteria and resume, not just one that has a similar title to what they’ve already applied to.

# Running Our Model

To run our model, there are two options: 

1. Run our train and test functions to see how our model works with our data. This will also run our evaluation feature and To do this, run our main.py file in this form:

python3 main.py --sentence "default"

2. Run the train function and then have the model find job matches for you. This will not run the evaluations but will run the training function and test function to find you matches. To do this, run main.py in this form:

python3 main.py --sentence "YOUR QUALIFIICATIOS AND EXPERIECES IN QUOTES" --n "Number of results you wish to see (don't put in quotes)

<img width="1146" alt="Screen Shot 2023-05-06 at 8 06 49 PM" src="https://user-images.githubusercontent.com/91433035/236651231-f0261b11-2e64-438b-95b6-715f92121878.png">

NOTE: Depending on your machine, our model takes quite a long time to run all the way. In case you don't want to sit through iterations of using our model, we've included some data from our tests.

Running an exact job description to find the most relevant:

<img width="676" alt="Screen Shot 2023-05-06 at 7 36 06 PM" src="https://user-images.githubusercontent.com/91433035/236650492-034b84b9-c22b-4095-9343-329341a2258e.png">


To run our Pytests navigate to the test directory and run

pytest tests.py

# Website Interface

We've made a user interface using pythonanywhere.com. To try out our model, navigate to anly521ling472aez.pythonanywhere.com. Enter in a description as described above, and you'll receive a list of sorted recommendations for jobs you should apply to! Happy job hunting!!



