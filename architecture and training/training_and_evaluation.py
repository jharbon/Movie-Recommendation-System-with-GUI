# -*- coding: utf-8 -*-

from os.path import isfile, join

import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from timeit import default_timer
from IPython.display import clear_output

def train(model, x_train, y_train, batch_size, epochs, save_name,
          save_model_path, history_path = None, lr = 0.001, lr_decay = True):
    
    if history_path is not None:
        if isfile(join(history_path, save_name)):
            return
    
    model.compile(loss = BinaryCrossentropy(), optimizer = Adam(learning_rate = lr), metrics = ["accuracy"])

    callback_list = []
    
    if history_path is not None:
        history_csv = CSVLogger(join(history_path, save_name))
        callback_list.append(history_csv)
    
    if lr_decay:
        lr_decay_callback = ReduceLROnPlateau(monitor = "loss",
                                 patience = 5,
                                 factor = 0.5,
                                 min_lr = 0.000001)
        callback_list.append(lr_decay_callback)
        
    if len(callback_list) == 0:
        model.fit(x = x_train, y = y_train, epochs = epochs, batch_size = batch_size)
    else:    
        model.fit(x = x_train, y = y_train, epochs = epochs, callbacks = callback_list, 
                  batch_size = batch_size)
    
    model.save(filepath = join(save_model_path, save_name + ".h5"), include_optimizer = False)

def hit_rate_top10(trained_model_path, test_set, total_set):
    trained_model = load_model(trained_model_path)
    
    test_user_movie_set = set(zip(test_set["user_ID"], test_set["movie_ID"]))
    
    user_seen_movies = total_set.groupby(by = "user_ID")["movie_ID"].apply(list).to_dict()
    
    movie_ID_set = set(total_set["movie_ID"])
    
    hits = []
    count = 0
    start = default_timer()
    for (user, movie) in test_user_movie_set:
        clear_output(wait = True)
        seen_movies = user_seen_movies[user]
        unseen_movies = movie_ID_set - set(seen_movies)
        
        movies_for_preds = list(np.random.choice(a = list(unseen_movies), size = 99))
            
        movies_for_preds.append(movie)
        
        preds = np.squeeze(trained_model.predict([np.array([user] * 100), np.array(movies_for_preds)]),
                               axis = 1)
        
        
        # Sort int_preds in ascending order in terms of indices
        # Reverse array order and select indices corresponding to movies in top 10 confidence values from trained_model.predict()
        # Use these indices to find movie_IDs for top 10 movie predictions
        top_10_movies = [movies_for_preds[i] for i in np.argsort(preds)[::-1][0:10].tolist()]
        
        # If the movie in this row of (user, movie) for the test set is in top 10 predictions, append 1 to hits to indicate success
        if movie in top_10_movies:
            hits.append(1)
        else:
            hits.append(0)
         
            
        count += 1 
        # Print percentage of loop completion and current time elapsed to predict total runtime for loop
        print("Overall Loop Progress: {:.2f}%".format((count/len(test_user_movie_set))*100))
        stop = default_timer()
        print("Current Overall Runtime: {:.2f} minutes".format((stop - start)/60)) 
        print("The Current Hit Rate is {:.2f}".format(np.average(hits) * 100))
            
    print("The Hit Rate Success for Top 10 Predictions is: {:.2f}%".format(np.average(hits) * 100)) 
        
        

    
    
    


    