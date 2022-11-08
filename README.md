# CDFW: Twitter sentiment analysis - Black Bears

The Wildlife Health Laboratory(WHL) at CDFW(California Department of Fish and Wildlife) includes a program that works to help resolve human-wildlife conflict in the state, including livestock depredation, property damage, public safety, and animal welfare issues. A major focus of this work is a human conflict with black bears. The State is working on updating its black bear conservation and management plan, therefore this research is a high priority for CDFW. 
USF(University of San Francisco) analysts would work with CDFW scientists(ecology & social science) to extract information from social media(primarily Twitter) using text analysis, sentiment analysis, and thematic coding.


# Objectives

The main focus of this project is to use social media data to detect the trend of encountering black bears in California and understand the public's view on human-wildlife interactions. 

We are currently working to achieve this by using machine learning to classify relevant tweets and perform sentiment analysis on them. Additionally, we plan to incorporate other covariates such as rainfall, temperate, and population density to see how they influence the number of occurrences and sentiment of human-wildlife interactions.


# Roadmap

## Scraping data 

Setting the developer environment and getting Twitter API; writing code to scrape data with keywords for selected periods; extracting features for each record; exploring demo for the automatic process(data pipeline).

- Scraping data
- Extracting features
- Building a pipeline

## Cleaning & Labeling 

Detecting the situations of the collected tweets (“encounter black bears or not”, “talking about real bears or not”); checking data types, deleting outliers, filling Nan values, etc.; manually category each record; creating a database.

## Training model 

Training the categorical model with the training set; tuning hyperparameters with the cross-validation sets; testing the model with the test set; evaluating the performance of various sentiment models.

## Visualization 

Deploying the model to production(AWS); building a dashboard for displaying the search results from the database; combining GIS data for further analyses; designing UI; monitoring and collecting data trends.

## Further exploration


# Collaborating reminder

* When you push your code to the repository, it might cause errors on other's machines if they don't have all the libraries you used. Please use the [twi.yml](https://github.com/persecond17/CDFW2023/blob/main/twi.yml) file in the repo to create/update the virtual environment for running programs on your end.

* A branch allows you to work independently of the master branch. After contributed to your branch, you can make a pull request to merge it with the master branch.

## Environment setting

First, pull twi.yml file:

`$ git pull`

Create or update an environment using twi.yml on your local machine:

`$ conda env create -f twi.yml -n [name_of_your_environment]`<br>
`$ conda env update -f twi.yml -n [name_of_your_environment]`

Activate or deactivate your environment on your local machine:

`$ conda activate [name_of_your_environment]`<br>
`$ conda deactivate`

Export the environment file to share and reproduce the current environments:

`$ conda env export > twi.yml`

Commit your contributions:

`$ git commit -a -m 'commit message'`

Push your updated environment onto github:

`$ git push origin`

## Creating a Branch

First, pull changes from upstream before creating a new branch:

`$ git pull`

Create a branch on your local machine and switch into this branch:

`$ git checkout -b [name_of_your_branch]`

Commit your contributions to this branch:

`$ git commit -a -m 'commit message'`

Push your branch onto github:

`$ git push origin [name_of_your_branch]`

Pushing your branch will create a pull request that can be reviewed before merging it with the master branch. You can follow the path **Pull requests -> New pull request -> (base: main, compared: your_branch) -> Create pull request** to create code review tasks on GitHub pages. 

If there is no conflict on files, you may click **Merge pull request -> Confirm merge -> Delete branch** to complete the merge process.