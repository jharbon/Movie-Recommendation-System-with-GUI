# Movie-Recommendation-System-with-GUI

I started this project because I wanted to improve my understanding of recommender systems such as those used by Netflix and Amazon, build my own models for predicting user interactions with movies and then create an application which someone can use to train a model and receive personalised movie recommendations.

NOTE: due to file size constraints, some files could not be included in this remote repo. These include: genome_scores.csv and ratings.csv, which were in the raw data folder; train_set, which was in the prepared dataset folder; and database.db, which was in the application data folder.

## Data

All of the raw data I used came from the MovieLens 25M dataset at https://grouplens.org/datasets/movielens/. The train and test sets were created from a dataset named ratings.csv, which contained 25M rows and each row was a review from a given user for a given movie, with information in columns "userId", "movieId", "rating" and "timestamp. Pandas and NumPy were used to clean this dataset, where I randomly selected a subset of the unique user set to reduce network training and evaluation runtime. Then, I transformed the data into an implicit feedback format, where 1 indicates an interaction and 0 indicates no interaction. To avoid a look-ahead bias, the leave-one-out test set methodology was used, where the pandas groupby object and the "timestamp" column were used to leave out the most recent review for each user and save it in a test set. Finally, the network I was planning to use needed to be exposed to rows with an interaction of 0, or "unseen" samples, as well as rows with an interaction of 1 if it is to successfully predict user interactions with movies. I created a function which could randomly select a movie which a given user has not seen, and append it to the training set with the user_ID, movie_ID and an interaction value of 0. The function repeated this process for every row in the dataset, and included a parameter called "ratio" which determined how many "unseen" samples were generated and appended for each "seen" sample. This transformation was only applied to the training set, and in the end this set contained approximately 40,000,000 rows and the test set contained over 50,000 rows. The size of the test set appears rather small in comparison to the training set, however this size is constrained by the nature of the evaluation metric used, which required a runtime of around 50 minutes for the aforementioned test set size.   

An extra file in the raw data folder named movies.csv was cleaned/transformed to create a dataset named movie_info, which was later used for the movie recommender application. The transformations involved: the creation of a new column named "year", where the year values were extracted from the movie titles; the renaming of one of the unique genre values; and the use of the pandas merge() function to perform an inner join (based on movie ID) on the movies dataset and the training datset, which ensured that the set of unique movie IDs was equal for both datasets.

Code and further details for the data cleaning and transformation process can be found in MovieLens_dataset_preparation.ipynb and movie_info_data_preparation.ipynb.

## Architecture and Training

Implemented the Neural Matrix Factorisation (NeuMF) architecture from He et al. (2017). This network first embeds users and movies into a latent vector space, and then propagates information through two separate paths, where one path applies the classical matrix factorisation method and the other is a multi-layer perceptron (MLP). The output vectors from the two separate paths are concatenated and then fed into a single output unit which is activated with a sigmoid function. The network was tuned on the prepared dataset via variation of the number of MLP units in each dense layer (two layers were used in my implementation), the batch size and the learning rate. 

## Hit Rate @ 10 Evaluation Metric

Accuracy could be used as the metric here, where we simply try to generalise our model to the test set and predict whether or not a given user will interact with a given movie. However, this metric is not very useful for real world uses of recommender systems, where we often need to decide on a subset of items to recommend to a given user by ranking items based on model confidence. For example, our model could technically achieve a high test accuracy even if most of its confidence predictions are borderline, i.e. confidence of 0.501 leading to a correct '1' interaction prediction or 0.499 leading to a correct '0' interaction prediction. For a better picture of how a recommender system might generalise for real world situations, the 'HitRate@10' metric is often used. 

For each user ID and movie ID pair in the test set, 99 movies which the given user has not seen are randomly selected and the movie in the test set which the user has seen is appended to this list to leave us with 100 movies. The trained model then predicts a confidence for the interaction of this user with these movies and ranks them based on the confidence values. The 10 movies with the highest confidence values for interaction are sliced from the list of predictions and then we check if the test set movie is in the top 10 list. If it is, we append a value of 1 to a list named 'hits', and if it is not then we append a value of 0. This process is repeated for every user-movie pair in the test set and at the end we can compute an average for the 'hits' list and label this the HitRate@10 score. After tuning the NeuMF network, a score of approximately 60% was achieved.

## Application 

The next goal was to create an application and GUI which could be used to create a user history, add the history to the training set, train the network with their new data and then receive personalised movie recommendations. For this task, I used python's tkinter library to create the GUI and used SQLite to create a database with various tables containing user, movie and training data. All of the classes and methods for the application are contained within application.py.

The first page is named LoginPage, and gives a user the option to select a user, create a user or edit a user profile.

![LoginPage](https://user-images.githubusercontent.com/73108695/125873738-87c69de0-c767-4d75-a17b-24afa5ba9c65.png)

The SelectUser page displays user profiles by listing the username (has to be unique for a given user) and the corresponding user ID. Users are given the option of entering a username and clicking 'Select User' to advance to the main application, or clicking 'Back' to return to the LoginPage.

![SelectUserPage](https://user-images.githubusercontent.com/73108695/125873981-4ab17728-c501-4f9c-a735-e94d97f2a844.png)

The CreateUser page simply gives the option for a user to create a new unique username. After entering the new username, the program connects to the database and checks in the user_info table to see if that username already exists. If it does not, then a new unique user ID is generated and a new row with the username and ID is appended to the table. For the case that there are no existing users in the database, the new user ID is set equal to the maximum ID value in the train_set table + 1. Otherwise, the new user ID is the highest ID in user_info + 1. 

The EditUserPage gives the user the option of changing the username of a profile or deleting a user profile. In the case that they delete a profile, all data for the corresponding is deleted from the user_info, user_history, user_bucket_list and train_set tables. 

![EditUserPage](https://user-images.githubusercontent.com/73108695/125874585-00323dc9-d902-405e-a9f9-bc2c9b0fa0f6.png)

The HomePage gives the user some basic information about how the application works. Menu buttons are distributed along the top of all of the main application pages, alongside the username in the top left.

![HomePage](https://user-images.githubusercontent.com/73108695/125874725-048a234f-58e5-446d-bd27-d8d325c656cf.png)

The HistoryPage allows a user to first add movies to their personal history, which is stored as rows with their username in the user_history table, and later view or edit it. Additionally, upon creation of a user history, they are required to click 'Finish' which appends their data to the train_set table with their corresponding user ID. To improve the quality of personalised recommendations, a user is required to add at least 20 movies to their history and is encouraged to add more to further improve the quality.  

In order to access movie data, the movie_info.csv file was used upon database creation to make a table named movie_info. Using this table, 1000 randomly selected movies are displayed in a treeview widget and a given movie can be added to user history via entering the corresponding movie ID. If a user happens to scroll through all 1000 movies without finding at least 20 they have seen, they can simply click the History menu button to refresh the list with a new random selection.

Checks are put into place via SQLite to ensure that the user has finished the creation of their history before they can train the network. Additionally, the user can re-train the network after they have edited their history and added more movies.

![HistoryPage_1](https://user-images.githubusercontent.com/73108695/125874757-a7d20df1-6319-4c5e-bcef-6bc6376df0e8.png)

![HistoryPage_2](https://user-images.githubusercontent.com/73108695/125874921-29c4333d-e589-4ae6-af9d-13bf97de5d13.png)

![HistoryPage_3](https://user-images.githubusercontent.com/73108695/125874933-82f86f3d-6607-4897-86cb-68d3229c3c1e.png)

After training the NeuMF network on the new data for the given user along with the pre-existing training data, the user is now ready to head to the 'Recommend' page. Upon clicking the 'Recommend Movies' button on that page, 100 movies are randomly selected from the movie_info table. The trained NeuMF network is loaded from a folder in application data and predicts confidence values for the given user's interaction with each of the 100 movies. The movies with the 10 highest confidence values are then displayed in the treeview widget below. The user has the option to add a movie to their bucket list via entering the corresponding movie ID. If they are not interested in any of the 10 movies, they can click 'Recommend Movies' again to repeat the process with a fresh list. 

![RecommendPage_1](https://user-images.githubusercontent.com/73108695/125874942-f1160879-3a7f-4585-802c-a955082c50a2.png)

![ReccomendPage_2](https://user-images.githubusercontent.com/73108695/125874955-d392c86d-1d94-47f8-8b50-f1f6faf5978c.png)

The BucketListPage displays all of the movies the user has added to their bucket list after receiving recommendations. The user can either remove a movie if they are no longer interested or they can add a movie to their history if they have now seen it. After addition of enough movies, they could then decide to re-train NeuMF to improve the quality of their future recommendations.

![BucketListPage](https://user-images.githubusercontent.com/73108695/125874964-1fc5143f-1945-4b38-a00a-baaa44e47e0c.png)









