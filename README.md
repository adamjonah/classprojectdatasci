# classprojectdatasci

Instructions For Running
To successfully get output, ensure that you download all of the files in the zip and have them in the correct placement.

First, you need to install dependencies. So in a virtual machine, run pip install -r requirements.txt. This will allow you to have all of the dependencies for the project.
Second, run the web scraper to get a pandas.DataFrame containing the latest news from all of the websites listed in the web scraper and of all of the tickers of interest (e.g. Apple, Coty, ExxonMobil)
Then, run the web scraper file again to get it ready for the keyword extraction. Thus, you have to run python news_scraper.py twice
To get the keyword extraction, type python KeywordDatabasePipeline.py ALGO N K with ALGO denoting the keyword extraction algorithm (YAKE, RAKE, TF-IDF, IDF), N denoting the number of sentences to look at from the scraped article and K denoting the number of keywords to extract from the sentences.
This will generate the output for the extracted keywords in ready_keyword_database.csv
For the keyword to text NN model, run python train_key_to_text.py to get sample output.
Enjoy!
