# -*- coding: utf-8 -*-

import tkinter as tk 
from tkinter import messagebox
from tkinter import ttk
import tkinter.font as font
from PIL import Image, ImageTk
    
import numpy as np
import pandas as pd    
import sqlite3 as sql
import requests    

from os import chdir, remove
from os.path import dirname, abspath, isfile

# Set working directory to folder which contains this script
chdir(dirname(abspath(__file__)))

# Append folder with training and architecture scripts to working directory
import sys
sys.path.append("architecture and training")
# Import NeuMF architecture and relevant functions for training
from NeuMF_architecture import NeuMF
from training_and_evaluation import train
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

from config import RAPID_API_KEY

# Set values for window dimensions here since some frames will be resized based on these dimensions
WINDOW_WIDTH = 1150
WINDOW_HEIGHT = 700

# Set constant value for border width of buttons
BTN_BORD_WIDTH = 3    

# Define this before running program to avoid long runtime when creating a user for first time
# This value will change depending on the train_set used for the database
TRAIN_SET_MAX_USER_ID = 162541

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Movie Recommender")
        self.iconbitmap("application data/images/popcorn_icon.ico")
        self.geometry(str(WINDOW_WIDTH) + "x" + str(WINDOW_HEIGHT))
        
        # Create a default font for buttons
        self.button_font = font.Font(family='Helvetica', size=9, weight='bold')
        
        self._frame = None
        self.change_frame(LoginPage)
        
        # Create master class attributes which can store key user information after login
        # Values can be accessed and modified in other classes
        self.username = None
        self.user_ID = None
        self.trained = None
        
        # Check if database already exists and create it if not
        if not isfile("application data/database.db"):
            self.create_database()    
            
    # Create a function for creating the database from scratch
    def create_database(self):
        # Create a database or connect to one
        conn = sql.connect("application data/database.db")
    
        # Create cursor
        c = conn.cursor()
            
        # Create tables
        c.executescript("""CREATE TABLE user_info(
                           username VARCHAR(20) PRIMARY KEY,
                           user_ID INT,
                           trained INT
                           );
                     
                           CREATE TABLE user_history(
                           username VARCHAR(20),
                           user_ID INT,
                           movie_ID INT
                           );
                           
                           CREATE TABLE user_bucket_list(
                           username VARCHAR(20),
                           user_ID INT,
                           movie_ID INT
                           );""")
            
        # Read train_set csv file in as chunks
        train_set_chunks = pd.read_csv("data/data preparation/dataset frac=0.33, ratio=4, min_samples=100/train_set.csv",
                                       chunksize = 50000, iterator=True)
            
        # Read chunks of csv into a new SQL table
        batch_num = 1
        for chunk in train_set_chunks:
            chunk.to_sql(name = "train_set", con = conn, index = False, if_exists='append')
            batch_num += 1
            print("Index number: {}".format(batch_num))
            
        c.execute("ALTER TABLE train_set DROP COLUMN 'Unnamed: 0'")
        c.execute("ALTER TABLE train_set ADD PRIMARY KEY (user_ID, movie_ID)")    
            
        # Read movie_info csv file in as chunks
        movie_info_chunks = pd.read_csv("data/data preparation/dataset frac=0.33, ratio=4, min_samples=100/movie_info.csv",
                                       chunksize = 1000, iterator=True)
            
        # Read chunks of csv into a new SQL table
        batch_num = 1
        for chunk in movie_info_chunks:
            chunk.to_sql(name = "movie_info", con = conn, index = False, if_exists='append')
            batch_num += 1
            print("Index number: {}".format(batch_num))    
            
        # Commit changes
        conn.commit()
        
        # Close connection
        conn.close()
        
    # Create a function for changing between various frames packed into App class
    def change_frame(self, frame_class):
        new_frame = frame_class(self)
        
        if self._frame is not None: 
            self._frame.destroy()
            
        self._frame = new_frame
        self._frame.pack()
    
    # Create a function for logging a user out and returning to the login page
    def logout(self):
        # Reset username and user_ID to None
        self.username = None
        self.user_ID = None
        
        # Take user back to login page
        self.change_frame(LoginPage)
        
        
class LoginPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        
        # Text which welcomes a user
        welcome_label = tk.Label(self, text = "Welcome to the movie recommender!")
        welcome_label.grid(row = 0, column = 0, columnspan = 2, pady = (150, 10))

        # Text which instructs user what to do
        welcome_label = tk.Label(self, text = 
                                 "Please select your user name\n" 
                                 "or create a new user profile")
        welcome_label.grid(row = 1, column = 0, columnspan = 2, pady = 10)                 
        
        # Add buttons for user select, user create or user edit
        
        user_select_button = tk.Button(self, text = "Select User", borderwidth = BTN_BORD_WIDTH,
                                       width = 30, bg = "grey", font = master.button_font,
                                       command = lambda: master.change_frame(SelectUserPage))
        user_select_button.grid(row = 2, column = 0, columnspan = 2, pady = 10)
        
        user_create_button = tk.Button(self, text = "Create User", borderwidth = BTN_BORD_WIDTH,
                                       width = 30, bg = "grey", font = master.button_font,
                                       command = lambda: master.change_frame(CreateUserPage))
        user_create_button.grid(row = 3, column = 0, columnspan = 2, pady = 10)
        
        user_edit_button = tk.Button(self, text = "Edit User", borderwidth = BTN_BORD_WIDTH,
                                       width = 30, bg = "grey", font = master.button_font,
                                       command = lambda: master.change_frame(EditUserPage))
        user_edit_button.grid(row = 4, column = 0, columnspan = 2, pady = (10, 150))
        
        
        # Configure frame to horizontally
        
        self.grid_columnconfigure(0, weight = 1)
        self.grid_columnconfigure(1, weight = 1)
        
class SelectUserPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        
        # Create label which instructs user how to select their profile
        instruct_label = tk.Label(self, text = "Enter your username from the records below")
        instruct_label.grid(row = 0, column = 0, columnspan = 2, pady = (120, 10))
        
        # Create text entry box with which the user can enter their username
        username_entry_label = tk.Label(self, text = "Username").grid(row = 1, column = 0, padx = 5, pady = 10)
        self.username_entry = tk.Entry(self, width = 30)
        self.username_entry.grid(row = 1, column = 1, pady = 10)
        
        # Store user records in a variable
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        c.execute("SELECT username, user_ID FROM user_info")
        records = c.fetchall()
        
        conn.close()
        
        # Create a frame to pack a Treeview widget into
        tree_frame = tk.Frame(self, width = 300, height = round(WINDOW_HEIGHT/2))
        tree_frame.grid(row = 2, column = 0, columnspan = 2, pady = 15)
        
        # Create a scrollbar for the frame
        tree_scroll = tk.Scrollbar(tree_frame)
        tree_scroll.pack(side = tk.RIGHT, fill = tk.Y)
        
        # Create a Treeview widget for displaying user records
        
        users_tree = ttk.Treeview(tree_frame, yscrollcommand = tree_scroll.set)
        users_tree.pack()
        
        tree_scroll.config(command = users_tree.yview)
        
        users_tree["columns"] = ("Username", "User ID")
        
        users_tree.column("#0", width = 0, minwidth = 0)
        users_tree.column("Username", anchor = "w", width = 300)
        users_tree.column("User ID", anchor = tk.CENTER, width = 120)
        
        users_tree.heading("#0", text = "")
        users_tree.heading("Username", text = "Username", anchor = tk.CENTER)
        users_tree.heading("User ID", text = "User ID", anchor = tk.CENTER)
        
        count = 0
        for record in records:
            users_tree.insert(parent = "", index = "end", iid = count, text = "", values = (record[0], record[1]))
            count += 1
        
        # Create a button which can be used to select a user once username has been entered
        select_button = tk.Button(self, text = "Select User", width = 30, borderwidth = BTN_BORD_WIDTH,
                                  bg = "grey", font = master.button_font,
                                  command = lambda: self.select_user(master))
        select_button.grid(row = 3, column = 0, columnspan = 2, pady = (5, 5))
        
        # Create a back button
        back_button = tk.Button(self, text = "Back", width = 30, borderwidth = BTN_BORD_WIDTH,
                                bg = "grey", font = master.button_font,
                                command = lambda: master.change_frame(LoginPage))
        back_button.grid(row = 4, column = 0, columnspan = 2, pady = (5, 120))
        
        # Configure frame to horizontally centre widgets
        self.grid_columnconfigure(0, weight = 1)
        self.grid_columnconfigure(1, weight = 1)
        
        
    # Create function which can select a user if they exist in the records
    def select_user(self, master):
        username = self.username_entry.get()
        if username == "":
            messagebox.showerror(title = "Invalid Username",
                                 message = "You did not enter a username!")
            return
            
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
            
        c.execute("SELECT username, user_ID, trained FROM user_info WHERE username = ?", (username,))
        records = c.fetchall()
        
        if len(records) == 0:
            messagebox.showerror(title = "Invalid Username", message = "That username does not exist!")   
        
            conn.close()
            return
        
        else:
            master.username = username
            master.user_ID = records[0][1]
            master.trained = records[0][2]
                    
            conn.close()
                    
            master.change_frame(HomePage)
            
                
class CreateUserPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)  
        
        # Create label which instructs user how to create their profile
        instruct_label = tk.Label(self, text = "Please create a username")
        instruct_label.grid(row = 0, column = 0, columnspan = 2, padx = (20, 30), pady = (220, 10))
        
        # Create text entry box with which the user can enter their chosen username
        username_entry_label = tk.Label(self, text = "Username")
        username_entry_label.grid(row = 1, column = 0, padx = 6)
        self.username_entry = tk.Entry(self, width = 30)
        self.username_entry.grid(row = 1, column = 1, padx = 6)
        
        # Create a button which can be used to select a user once username has been entered
        create_button = tk.Button(self, text = "Create User", width = 30, borderwidth = BTN_BORD_WIDTH,
                                  bg = "grey", font = master.button_font,
                                  command = lambda: self.create_user(master))
        create_button.grid(row = 2, column = 0, columnspan = 2, pady = (15, 5))
        
        # Create a back button
        back_button = tk.Button(self, text = "Back", width = 30, borderwidth = BTN_BORD_WIDTH,
                                bg = "grey", font = master.button_font,
                                command = lambda: master.change_frame(LoginPage))
        back_button.grid(row = 3, column = 0, columnspan = 2, pady = (5, 220))
        
        # Configure frame to horizontally centre widgets
        self.grid_columnconfigure(0, weight = 1)
        self.grid_columnconfigure(1, weight = 1)
        
    def create_user(self, master):
        username = self.username_entry.get()
            
        if username == "":
            messagebox.showerror(title = "Invalid Username", 
                                 message = "You cannot enter an empty username!")
            self.username_entry.delete(0, tk.END)
            return
            
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
            
        c.execute("SELECT username FROM user_info")
        records = c.fetchall()
        
        conn.close()
            
        for record in records:
            if record[0] == username:
                messagebox.showerror(title = "Invalid Username",
                                     message = "That username is already taken!")
                    
                self.username_entry.delete(0, tk.END)
                return
            
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        if len(records) == 0:
            c.execute("INSERT INTO user_info VALUES (:username, :user_ID, :trained)",
                     {
                     "username": username,
                     "user_ID": TRAIN_SET_MAX_USER_ID + 1,
                     "trained": 0
                     })
            
            master.username = username
            master.user_ID = TRAIN_SET_MAX_USER_ID + 1
            
        else:
            c.execute("SELECT MAX(user_ID) FROM user_info")
            max_user_ID = c.fetchall()[0][0]
            
            c.execute("INSERT INTO user_info VALUES (:username, :user_ID, :trained)",
                     {
                     "username": username,
                     "user_ID": max_user_ID + 1,
                     "trained": 0
                     })
            
            master.username = username
            master.user_ID = max_user_ID + 1
        
        conn.commit()
        conn.close()
                    
        
        self.username_entry.delete(0, tk.END)
        
        master.change_frame(HomePage)
            
        
class EditUserPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)  
        
        # Create label which instructs user how to delete their profile
        instruct_label = tk.Label(self, text = "Enter the username for the profile " + 
                                               "you wish to edit. You can then either\n" +
                                               "enter a new username and submit or you " +
                                               "can simply click delete to\n" +
                                               "remove your user profile from the records.")
        instruct_label.grid(row = 0, column = 0, columnspan = 2, pady = (70, 10))
        
        # Create text entry box with which the user can enter their ID
        username_entry_label = tk.Label(self, text = "Username").grid(row = 1, column = 0, padx = 3, pady = (15, 5))
        self.username_entry = tk.Entry(self, width = 30)
        self.username_entry.grid(row = 1, column = 1, pady = (15, 5))
        
        # Create text entry box with which the user can enter their new username
        new_username_entry_label = tk.Label(self, text = "New Username").grid(row = 2, column = 0, padx = 3, pady = (5, 15))
        self.new_username_entry = tk.Entry(self, width = 30)
        self.new_username_entry.grid(row = 2, column = 1, pady = (5, 15))
        
        # Store user records in a variable
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        c.execute("SELECT * FROM user_info")
        records = c.fetchall()
        
        conn.close()
        
        # Create a frame to pack a Treeview widget into
        tree_frame = tk.Frame(self, width = 300, height = round(WINDOW_HEIGHT/2))
        tree_frame.grid(row = 3, column = 0, columnspan = 2, pady = 15)
        
        # Create a scrollbar for the frame
        tree_scroll = tk.Scrollbar(tree_frame)
        tree_scroll.pack(side = tk.RIGHT, fill = tk.Y)
        
        # Create a Treeview widget for displaying user records
        
        users_tree = ttk.Treeview(tree_frame, yscrollcommand = tree_scroll.set)
        users_tree.pack()
        
        tree_scroll.config(command = users_tree.yview)
        
        users_tree["columns"] = ("Username", "User ID")
        
        users_tree.column("#0", width = 0, minwidth = 0)
        users_tree.column("Username", anchor = "w", width = 300)
        users_tree.column("User ID", anchor = tk.CENTER, width = 120)
        
        users_tree.heading("#0", text = "")
        users_tree.heading("Username", text = "Username", anchor = tk.CENTER)
        users_tree.heading("User ID", text = "User ID", anchor = tk.CENTER)
        
        count = 0
        for record in records:
            users_tree.insert(parent = "", index = "end", iid = count, text = "", values = (record[0], record[1]))
            count += 1
        
        # Create a button which can be used to change username once original and new have been entered
        change_username_button = tk.Button(self, text = "Change Username", width = 30, borderwidth = BTN_BORD_WIDTH,
                                           bg = "grey", font = master.button_font,
                                           command = lambda: self.change_username(master))
        change_username_button.grid(row = 4, column = 0, columnspan = 2, pady = (5, 5))
        
        # Create a button which can be used to delete a user profile once a username has been entered
        delete_user_button = tk.Button(self, text = "Delete User", width = 30, borderwidth = BTN_BORD_WIDTH,
                                       bg = "grey", font = master.button_font,
                                command = lambda: self.delete_user(master))
        delete_user_button.grid(row = 5, column = 0, columnspan = 2, pady = 5)
        
        # Create a back button
        back_button = tk.Button(self, text = "Back", width = 30, borderwidth = BTN_BORD_WIDTH,
                                bg = "grey", font = master.button_font,
                                command = lambda: master.change_frame(LoginPage))
        back_button.grid(row = 6, column = 0, columnspan = 2, pady = (5, 80))
        
        # Configure frame to horizontally centre widgets
        self.grid_columnconfigure(0, weight = 1)
        self.grid_columnconfigure(1, weight = 1)
        
    def change_username(self, master):
        username = self.username_entry.get()
        new_username = self.new_username_entry.get()
        if username == "" or new_username == "":
            messagebox.showerror(title = "Invalid Username",
                                 message = "You must enter a username in both text boxes if you want to edit your username!")
            return
            
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
            
        c.execute("SELECT * FROM user_info WHERE username = ?", (username,))
        records = c.fetchall()
        if len(records) == 0:
            messagebox.showerror(title = "Invalid Username", 
                                 message = "That username does not exist!")
            
            conn.close()
            return
            
        c.execute("SELECT * FROM user_info WHERE username = ?", (new_username,))
        records = c.fetchall()
        if len(records) != 0:
            messagebox.showerror(title = "Invalid New Username", 
                                 message = "The new username already exists!")  
            
            conn.close()
            return
                    
        conn.close()
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        c.execute("""UPDATE user_info SET username = ?
                     WHERE username = ?""", (new_username, username))
                     
        c.execute("""UPDATE user_history SET username = ?
                     WHERE username = ?""", (new_username, username)) 
                     
        c.execute("""UPDATE user_bucket_list SET username = ?
                     WHERE username = ?""", (new_username, username))               
        
        conn.commit()
        conn.close()
        
        messagebox.showinfo(title = "Username Updated",
                            message = "You have successfully updated your username!")
        
        master.change_frame(LoginPage)
        
    def delete_user(self, master):
        username = self.username_entry.get()
        
        if username == "":
            messagebox.showerror(title = "Invalid Username",
                                 message = "You must enter a username!")
            return
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
            
        c.execute("SELECT username, user_ID FROM user_info WHERE username = ?", (username,))
        records = c.fetchall()
        if len(records) == 0:
            messagebox.showerror(title = "Invalid Username", 
                                 message = "That username does not exist!")
            
            conn.close()
            return
        
        conn.close()
        
        user_ID = records[0][1]
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        c.execute("DELETE FROM user_info WHERE username = ?", (username,))
        c.execute("DELETE FROM user_history WHERE username = ?", (username,))
        c.execute("DELETE FROM user_bucket_list WHERE username = ?", (username,))
        c.execute("DELETE FROM train_set WHERE user_ID = ?", (user_ID,))
        
        conn.commit()
        conn.close()
        
        messagebox.showinfo(title = "User Deleted",
                            message = "You have successfully deleted the profile with username: " + str(username))
        
        master.change_frame(LoginPage)
        
                        

# Define constant properties for the menu buttons
MENU_BTN_PADX = 8
MENU_BTN_PADY = 10
MENU_BTN_WIDTH = 20
MENU_BTN_HEIGHT = 2        
        
class HomePage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master) 
        
        # Define buttons for switching between pages after user login or creation
        # Username displayed to the left of buttons
        
        username_display = tk.Label(self, text = master.username, font = master.button_font,)
        username_display.grid(row = 0, column = 0, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        home_button = tk.Button(self, text = "Home", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                command = lambda: master.change_frame(HomePage))
        home_button.grid(row = 0, column = 1, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        bucket_list_button = tk.Button(self, text = "Bucket List", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                       borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                       command = lambda: master.change_frame(BucketListPage))
        bucket_list_button.grid(row = 0, column = 2, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        history_button = tk.Button(self, text = "History", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                   borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                   command = lambda: master.change_frame(HistoryPage))
        history_button.grid(row = 0, column = 3, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
    
        recommend_button = tk.Button(self, text = "Recommend", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                     borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                     command = lambda: master.change_frame(RecommendPage))
        recommend_button.grid(row = 0, column = 4, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        logout_button = tk.Button(self, text = "Logout", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                  borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                  command = lambda: master.change_frame(LoginPage))
        logout_button.grid(row = 0, column = 5, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        info_label = tk.Label(self,
                              text = "Hello " + master.username + "! Welcome to the movie recommendation system. Before " +
                                     "you can ask the system to recommend movies, you must first head " +
                                     "over to the History page and tell us what you have already seen. Once you " +
                                     "have added some movies, we'll save your history in our database and it will " +
                                     "be added to the training set after you have clicked 'Finish'.\n\n"
                                     "To achieve the best results, NeuMF is trained on approximately 40 million " +
                                     "rows of data along with the data you provide. This means that, with the training " +
                                     "configurations we have decided on, you will have to wait approximately 15 minutes " +
                                     "for NeuMF to finish training. To start the training, head to the History page and " +
                                     "click the 'Train' button. Upon finishing the training, you can go to the " +
                                     "Recommend page, optionally select a genre and click 'Recommend Movies'. Bucket List " +
                                     "displays movies stored in your own personal bucket list. Enjoy!",
                                     wraplength = round(WINDOW_WIDTH/2))
        
        info_label.grid(row = 1, column = 0, columnspan = 6, pady = 15)
        
        # Create label and use it to display a movie image
        
        resolution = (600, 300)
        self.movie_image = Image.open("application data/images/the_good_the_bad_and_the_ugly_movie.jpg").resize(resolution)
        self.movie_image = ImageTk.PhotoImage(self.movie_image)
        movie_image_label = tk.Label(self, image = self.movie_image)
        movie_image_label.grid(row = 2, column = 0, columnspan = 6, pady = 15)
        
        self.grid_columnconfigure(0, weight = 1)
        self.grid_columnconfigure(1, weight = 1)
        self.grid_columnconfigure(2, weight = 1)
        self.grid_columnconfigure(3, weight = 1)
        self.grid_columnconfigure(4, weight = 1)
        self.grid_columnconfigure(5, weight = 1)
        self.grid_columnconfigure(6, weight = 1)

class BucketListPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        
        # Define buttons for switching between pages after user login or creation
        # Username displayed to the left of buttons
        
        username_display = tk.Label(self, text = master.username, font = master.button_font,)
        username_display.grid(row = 0, column = 0, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        home_button = tk.Button(self, text = "Home", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                command = lambda: master.change_frame(HomePage))
        home_button.grid(row = 0, column = 1, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        bucket_list_button = tk.Button(self, text = "Bucket List", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                       borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                       command = lambda: master.change_frame(BucketListPage))
        bucket_list_button.grid(row = 0, column = 2, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        history_button = tk.Button(self, text = "History", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                   borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                   command = lambda: master.change_frame(HistoryPage))
        history_button.grid(row = 0, column = 3, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        recommend_button = tk.Button(self, text = "Recommend", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                     borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                     command = lambda: master.change_frame(RecommendPage))
        recommend_button.grid(row = 0, column = 4, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        logout_button = tk.Button(self, text = "Logout", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                  borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                  command = lambda: master.change_frame(LoginPage))
        logout_button.grid(row = 0, column = 5, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        # Create text which instructs user how to use this page
        info_label = tk.Label(self, 
                              text = "This page displays all of the recommended movies you have decided to add to your " +
                              "personal bucket list! To edit a movie in the list, you can enter its corresponding ID " +
                              "and then choose to either delete it if you are no longer interested, or add it to " +
                              "your history if you have now seen it. Upon addition to your history, NeuMF can be retrained.",
                              wraplength = round(WINDOW_WIDTH/2))
        info_label.grid(row = 1, column = 2, columnspan = 3, pady = 20)
        
        # Create movie ID label and text entry box
        movie_ID_entry_label = tk.Label(self, text = "Movie ID")
        movie_ID_entry_label.grid(row = 2, column = 2, columnspan = 2, padx = (0, 7), pady = 15)
        self.movie_ID_text_entry = tk.Entry(self, width = 20)
        self.movie_ID_text_entry.grid(row = 2, column = 3, columnspan = 2, padx = (7, 5), pady = 15)
        
        # Create a button for removing a movie
        delete_button = tk.Button(self, text = "Delete Movie", width = 20, borderwidth = BTN_BORD_WIDTH,
                                  bg = "grey", font = master.button_font,
                                  command = lambda: self.delete_movie(master))
        delete_button.grid(row = 3, column = 2, columnspan =  2, pady = 10)
        
        # Create a button for adding a movie to history
        add_history_button = tk.Button(self, text = "Add to History", width = 20, borderwidth = BTN_BORD_WIDTH,
                                       bg = "grey", font = master.button_font,
                                       command = lambda: self.add_to_history(master))
        add_history_button.grid(row = 3, column = 3, columnspan = 2, pady = 10)
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        c.execute("""SELECT title, genre, year, movie_info.movie_ID FROM movie_info
                  INNER JOIN user_bucket_list
                  ON movie_info.movie_ID = user_bucket_list.movie_ID
                  WHERE user_bucket_list.username = ?
                  ORDER BY title""", (master.username,))
                  
        records = c.fetchall()
        
        conn.close()
        
        # Create a frame to pack a Treeview widget into
        tree_frame = tk.Frame(self, width = 1000, height = round(WINDOW_HEIGHT/2))
        tree_frame.grid(row = 4, column = 0, columnspan = 6, pady = 15)
        
        # Create a scrollbar for the frame
        tree_scroll = tk.Scrollbar(tree_frame)
        tree_scroll.pack(side = tk.RIGHT, fill = tk.Y)
        
        # Create a Treeview widget for displaying user records
        
        bucket_list_tree = ttk.Treeview(tree_frame, yscrollcommand = tree_scroll.set)
        bucket_list_tree.pack()
        
        tree_scroll.config(command = bucket_list_tree.yview)
        
        bucket_list_tree["columns"] = ("Title", "Genre", "Year", "Movie ID")
        
        bucket_list_tree.column("#0", width = 0, minwidth = 0)
        bucket_list_tree.column("Title", anchor = "w", width = 450)
        bucket_list_tree.column("Genre", anchor = "w", width = 350)
        bucket_list_tree.column("Year", anchor = tk.CENTER, width = 100)
        bucket_list_tree.column("Movie ID", anchor = tk.CENTER, width = 100)
        
        bucket_list_tree.heading("#0", text = "")
        bucket_list_tree.heading("Title", text = "Title", anchor = tk.CENTER)
        bucket_list_tree.heading("Genre", text = "Genre", anchor = tk.CENTER)
        bucket_list_tree.heading("Year", text = "Year", anchor = tk.CENTER)
        bucket_list_tree.heading("Movie ID", text = "Movie ID", anchor = tk.CENTER)
        
        count = 0
        for record in records:
            bucket_list_tree.insert(parent = "", index = "end", iid = count, text = "",
                                    values = (record[0], record[1], record[2], record[3]))
            count += 1
        
        
    def check_ID(self, master):
        movie_ID = self.movie_ID_text_entry.get()
        
        if movie_ID == "":
            messagebox.showerror(title = "Invalid Movie ID", message = "You did not enter a movie ID!")
            return False
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
            
        c.execute("SELECT username, movie_ID FROM user_bucket_list WHERE username = ? AND movie_ID = ?",
                  (master.username, int(movie_ID)))
        records = c.fetchall()
        if len(records) == 0:
            messagebox.showerror(title = "Invalid Movie ID", 
                                 message = "That movie ID is not in your bucket list!")
            
            conn.close()
            return False
        
        return True
        
    def delete_movie(self, master):
        ID_valid = self.check_ID(master)
        
        if not ID_valid:
            return
        
        movie_ID = self.movie_ID_text_entry.get()
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        c.execute("SELECT title FROM movie_info WHERE movie_ID = ?", (int(movie_ID),))
        movie_title = c.fetchall()[0][0]
        
        c.execute("DELETE FROM user_bucket_list WHERE username = ? AND movie_ID = ?",
                  (master.username, int(movie_ID)))
        
        conn.commit()
        conn.close()
        
        messagebox.showinfo(title = "Movie Removed",
                            message = "You have removed " + "'" + movie_title + "'")
        
        master.change_frame(BucketListPage)
    
    def add_to_history(self, master):
        ID_valid = self.check_ID(master)
        
        if not ID_valid:
            return
        
        movie_ID = self.movie_ID_text_entry.get()
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        c.execute("SELECT title FROM movie_info WHERE movie_ID = ?", (int(movie_ID),))
        movie_title = c.fetchall()[0][0]
        
        c.execute("DELETE FROM user_bucket_list WHERE username = ? AND movie_ID = ?",
                  (master.username, int(movie_ID)))
        
        c.execute("INSERT INTO user_history VALUES (:username, :user_ID, :movie_ID)",
                  {
                     "username": master.username,
                     "user_ID": master.user_ID,
                     "movie_ID": int(movie_ID)
                  })
        
        c.execute("UPDATE user_info SET trained = 0 WHERE username = ?",
                  (master.username,))
        
        conn.commit()
        conn.close()
        
        messagebox.showinfo(title = "Movie Added",
                            message = "You have added " + "'" + movie_title + "' " + "to your history")
        
        master.change_frame(BucketListPage)
        
class HistoryPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master) 
        
        # Define buttons for switching between pages after user login or creation
        # Username displayed to the left of buttons
        
        username_display = tk.Label(self, text = master.username, font = master.button_font,)
        username_display.grid(row = 0, column = 0, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        home_button = tk.Button(self, text = "Home", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                command = lambda: master.change_frame(HomePage))
        home_button.grid(row = 0, column = 1, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        bucket_list_button = tk.Button(self, text = "Bucket List", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                       borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                       command = lambda: master.change_frame(BucketListPage))
        bucket_list_button.grid(row = 0, column = 2, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        history_button = tk.Button(self, text = "History", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                   borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                   command = lambda: master.change_frame(HistoryPage))
        history_button.grid(row = 0, column = 3, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        recommend_button = tk.Button(self, text = "Recommend", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                     borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                     command = lambda: master.change_frame(RecommendPage))
        recommend_button.grid(row = 0, column = 4, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        logout_button = tk.Button(self, text = "Logout", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                  borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                  command = lambda: master.change_frame(LoginPage))
        logout_button.grid(row = 0, column = 5, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)

        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        # Check how many existing records are in this user's history
        c.execute("SELECT * FROM user_history WHERE username = ?", (master.username,))
        records = c.fetchall()
        conn.close()
        
        # Create label and text entry box for movie IDs
        self.movie_ID_entry_label = tk.Label(self, text = "Movie ID")
        self.movie_ID_entry_label.grid(row = 2, column = 2, columnspan = 2, pady = 15)
        self.movie_ID_text_entry = tk.Entry(self, width = 20)
        self.movie_ID_text_entry.grid(row = 2, column = 3, columnspan = 2, pady = 15)
        
        # Create button for editing history
        self.edit_history_button = tk.Button(self, text = "Edit History", borderwidth = BTN_BORD_WIDTH,
                                        bg = "grey", font = master.button_font,
                                        width = 20, command = lambda: self.create_edit_history(master))
        self.edit_history_button.grid(row = 3, column = 2, padx = (0, 10), pady = 15)
        
        # Create button for deleting a movie from history and training set based on the entered movie ID
        self.delete_movie_button = tk.Button(self, text = "Delete Movie", borderwidth = BTN_BORD_WIDTH,
                                        bg = "grey", font = master.button_font,
                                        width = 20, command = lambda: self.delete_movie(master))
        self.delete_movie_button.grid(row = 3, column = 3, padx = (0, 10), pady = 15)
        
        # Create button for training NeuMF network
        self.train_NeuMF_button = tk.Button(self, text = "Train", borderwidth = BTN_BORD_WIDTH, width = 20,
                                       bg = "grey", font = master.button_font,
                                       command = lambda: self.train_NeuMF(master))
        self.train_NeuMF_button.grid(row = 3, column = 4, padx = (10, 0), pady = 15)
        
        # Create a frame to pack a Treeview widget into
        self.tree_frame = tk.Frame(self, width = 1000, height = round(WINDOW_HEIGHT/2))
        self.tree_frame.grid(row = 4, column = 0, columnspan = 7, pady = 15)
        
        # Create widgets for creation of user history if number of movies is less than 20
        if len(records) < 20:
            info_label = tk.Label(self,
                                  text = "It appears that you have not added enough movies to your history yet. Scroll " +
                                  "through the films below until you see a movie you have watched. Type the movie ID " +
                                  "into the text box and click 'Add Movie'. Repeat this process until you have added " +
                                  "at least 20 movies. The counter below keeps track of how many you have added so far. " +
                                  "Make sure that you click 'Finish' once you have selected at least 20 movies. As a " +
                                  "final note, if you reach the bottom of the movies list and have not added at least " +
                                  "20, you can simply click the 'History' button at the top and refresh the list.",
                                  wraplength = round(WINDOW_WIDTH/2))
            
            info_label.grid(row = 1, column = 2, columnspan = 3, pady = 20)
            self.create_edit_history(master, history_creation = True)
        
        # Display user history if it exists and number of movies is greater than or equal to 20   
        else:
            info_label = tk.Label(self, 
                              text = "This page displays all of the movies in your history. If you change your mind " +
                                      "and want to add more, you can click 'Edit History'. Additionally, you can delete a " +
                                      "movie from your history via its ID and 'Delete Movie'. Once you have added your " +
                                      "history to the training set with the 'Finish' button, you can click 'Train' and " +
                                      "wait until you see 'Training finished!' appear.",
                                      wraplength = round(WINDOW_WIDTH/2))
            info_label.grid(row = 1, column = 2, columnspan = 3, pady = 20)
            
            conn = sql.connect("application data/database.db")
            c = conn.cursor()
            
            c.execute("""SELECT title, genre, year, movie_info.movie_ID FROM movie_info
                      INNER JOIN user_history
                      ON movie_info.movie_ID = user_history.movie_ID
                      WHERE user_history.username = ?
                      ORDER BY title""", (master.username,))
                      
            records = c.fetchall()
            
            conn.close()
            
            # Create a scrollbar for the frame
            tree_scroll = tk.Scrollbar(self.tree_frame)
            tree_scroll.pack(side = tk.RIGHT, fill = tk.Y)
            
            # Create a Treeview widget for displaying user records
            
            history_tree = ttk.Treeview(self.tree_frame, yscrollcommand = tree_scroll.set)
            history_tree.pack()
            
            tree_scroll.config(command = history_tree.yview)
            
            history_tree["columns"] = ("Title", "Genre", "Year", "Movie ID")
            
            history_tree.column("#0", width = 0, minwidth = 0)
            history_tree.column("Title", anchor = "w", width = 450)
            history_tree.column("Genre", anchor = "w", width = 350)
            history_tree.column("Year", anchor = tk.CENTER, width = 100)
            history_tree.column("Movie ID", anchor = tk.CENTER, width = 100)
            
            history_tree.heading("#0", text = "")
            history_tree.heading("Title", text = "Title", anchor = tk.CENTER)
            history_tree.heading("Genre", text = "Genre", anchor = tk.CENTER)
            history_tree.heading("Year", text = "Year", anchor = tk.CENTER)
            history_tree.heading("Movie ID", text = "Movie ID", anchor = tk.CENTER)
            
            count = 0
            for record in records:
                history_tree.insert(parent = "", index = "end", iid = count, text = "",
                                        values = (record[0], record[1], record[2], record[3]))
                count += 1
    
    # Create function which creates the widgets for creation and editing of user history
    def create_edit_history(self, master, history_creation = False):
        # Widgets for viewing history could be present when this function is called
        # Make them invisible
        self.movie_ID_entry_label.grid_forget()
        self.movie_ID_text_entry.grid_forget()
        self.edit_history_button.grid_forget()
        self.delete_movie_button.grid_forget()
        self.train_NeuMF_button.grid_forget()
        self.tree_frame.grid_forget()
        
        if not history_creation:   
            info_label = tk.Label(self,
                                      text = "Scroll through the films below until you see a movie you have watched. Type the ID " +
                                      "into the text box and click 'Add Movie'. You can search for general or specific titles with " +
                                      "the 'Search Movie' button and movie title text box. The counter below keeps track of how many you have added so far. " +
                                      "Make sure that you click 'Finish' once you have selected at least 20 movies. As a " +
                                      "final note, if you reach the bottom of the movies list and have not added at least " +
                                      "20, you can simply click the 'History' button at the top to refresh the list.",
                                      wraplength = round(WINDOW_WIDTH/2))
                
            info_label.grid(row = 1, column = 2, columnspan = 3, pady = 20)
        
        # Create label and text entry box for movie IDs
        movie_ID_entry_label = tk.Label(self, text = "Movie ID")
        movie_ID_entry_label.grid(row = 2, column = 2, columnspan = 2, pady = (15, 5))
        self.movie_ID_text_entry = tk.Entry(self, width = 20)
        self.movie_ID_text_entry.grid(row = 2, column = 3, columnspan = 2, pady = (15, 5))
        
        # Create label and text entry box for searching movie titles
        movie_title_entry_label = tk.Label(self, text = "Movie Title")
        movie_title_entry_label.grid(row = 3, column = 2, columnspan = 2, pady = (5, 15))
        self.movie_title_text_entry = tk.Entry(self, width = 20)
        self.movie_title_text_entry.grid(row = 3, column = 3, columnspan = 2, pady = (5, 15))
        
        # Create button for searching a movie title and opening new window with treeview of search results
        search_movie_button = tk.Button(self, text = "Search Movie", width = 20, borderwidth = BTN_BORD_WIDTH,
                                     bg = "grey", font = master.button_font,
                                     command = lambda: self.search_title(master))
        search_movie_button.grid(row = 4, column = 2, pady = 10)
        
        # Create button for adding a movie to history based on the entered movie ID
        add_movie_button = tk.Button(self, text = "Add Movie", width = 20, borderwidth = BTN_BORD_WIDTH,
                                     bg = "grey", font = master.button_font,
                                     command = lambda: self.add_movie(master))
        add_movie_button.grid(row = 4, column = 3, pady = 10)
        
        # Create button for finishing the process of creating a user history
        finish_button = tk.Button(self, text = "Finish", width = 20, borderwidth = BTN_BORD_WIDTH,
                                  bg = "grey", font = master.button_font,
                                  command = lambda: self.finish(master))
        finish_button.grid(row = 4, column = 4, pady = 10)
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
    
        c.execute("""SELECT title, genre, year, movie_ID FROM movie_info
                     WHERE movie_ID IN (SELECT movie_ID FROM movie_info ORDER BY RANDOM() LIMIT 1000)""")
                  
        records = c.fetchall()
        conn.close()
        
        # Create a frame to pack a Treeview widget into
        tree_frame = tk.Frame(self, width = 1000, height = round(WINDOW_HEIGHT/2))
        tree_frame.grid(row = 5, column = 0, columnspan = 6, pady = 15)
        
        # Create a scrollbar for the frame
        tree_scroll = tk.Scrollbar(tree_frame)
        tree_scroll.pack(side = tk.RIGHT, fill = tk.Y)
        
        # Create a Treeview widget for displaying user records
        
        movies_tree = ttk.Treeview(tree_frame, yscrollcommand = tree_scroll.set)
        movies_tree.pack()
        
        tree_scroll.config(command = movies_tree.yview)
        
        movies_tree["columns"] = ("Title", "Genre", "Year", "Movie ID")
        
        movies_tree.column("#0", width = 0, minwidth = 0)
        movies_tree.column("Title", anchor = "w", width = 450)
        movies_tree.column("Genre", anchor = "w", width = 350)
        movies_tree.column("Year", anchor = tk.CENTER, width = 100)
        movies_tree.column("Movie ID", anchor = tk.CENTER, width = 100)
        
        movies_tree.heading("#0", text = "")
        movies_tree.heading("Title", text = "Title", anchor = tk.CENTER)
        movies_tree.heading("Genre", text = "Genre", anchor = tk.CENTER)
        movies_tree.heading("Year", text = "Year", anchor = tk.CENTER)
        movies_tree.heading("Movie ID", text = "Movie ID", anchor = tk.CENTER)
        
        count = 0
        for record in records:
            movies_tree.insert(parent = "", index = "end", iid = count, text = "",
                                    values = (record[0], record[1], record[2], record[3]))
            count += 1
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
    
        c.execute("SELECT * FROM user_history WHERE username = ?", (master.username,))
        records = c.fetchall()
        conn.close()
        
        # Create label below treeview box which tells user how many movies they have added so far
        self.movie_count_label = tk.Label(self, text = "Movies added: {}/20".format(len(records)))
        self.movie_count_label.grid(row = 6, column = 4, columnspan = 2, padx = (70, 0)) 
        
    # Create function for deleting a movie from user history    
    def delete_movie(self, master):
        movie_ID = self.movie_ID_text_entry.get()
        
        if movie_ID == "":
            messagebox.showerror(title = "Invalid Movie ID", message = "You did not enter a movie ID!")
            return
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        # Check if movie is in history
        c.execute("SELECT * FROM user_history WHERE username = ? AND movie_ID = ?",
                  (master.username, int(movie_ID)))
        records = c.fetchall()
        if len(records) == 0:
            messagebox.showerror(title = "Invalid Movie ID", 
                                 message = "That movie is not in your history!")
            
            conn.close()
            return
        
        # Check if entered movie ID exists in records
        c.execute("SELECT * FROM movie_info WHERE movie_ID = ?", (int(movie_ID),))
        records = c.fetchall()
        if len(records) == 0:
            messagebox.showerror(title = "Invalid Movie ID",
                                 message = "That movie ID does not exist in our records!")
            
            conn.close()
            return
        
        c.execute("SELECT title FROM movie_info WHERE movie_ID = ?", (int(movie_ID),))
        movie_title = c.fetchall()[0][0]
        
        # Display selected movie in messagebox and ask user if they are sure they want to add it
        decision = messagebox.askquestion(title = "Movie Selected",
                                          message = "You have decided to delete '" + movie_title + "' from your history. " +
                                          "Is this correct?")
        if decision == "yes":
            c.execute("DELETE FROM user_history WHERE user_ID = ? AND movie_ID = ?", (master.user_ID, int(movie_ID)))
            c.execute("DELETE FROM train_set WHERE user_ID = ? AND movie_ID = ?", (master.user_ID, int(movie_ID)))
            conn.commit()
            conn.close()
            
            master.change_frame(HistoryPage)
            
        else:
            return
    
    # Create function for searching movie_info table for titles equal or similar to entered title
    # New window created with search results in treeview widget
    def search_title(self, master):
        movie_title = self.movie_title_text_entry.get()
        
        if movie_title == "":
            messagebox.showerror(title = "Invalid Movie Title", message = "You did not enter a movie title!")
            return
        
        results_window = tk.Toplevel(master)
        results_window.title("Search Results")
        results_window.geometry(str(WINDOW_WIDTH) + "x" + str(round(WINDOW_HEIGHT/2)))
        
        info_label = tk.Label(results_window, text = "Here are the results we found for '" + movie_title + "'.")
        info_label.grid(row = 0, column = 0, columnspan = 2, padx = 60, pady = 15)
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        # Use 'LIKE' operator to find movie titles equal or similar to entered title
        c.execute("""SELECT title, genre, year, movie_ID FROM movie_info
                     WHERE title LIKE ?""", ("%" + movie_title + "%",))
                     
        records = c.fetchall()
        conn.close()
        
        # Create a frame to pack a Treeview widget into
        tree_frame = tk.Frame(results_window, width = 1000, height = round(WINDOW_HEIGHT/2))
        tree_frame.grid(row = 1, column = 0, columnspan = 2, padx = 60, pady = 15)
        
        # Create a scrollbar for the frame
        tree_scroll = tk.Scrollbar(tree_frame)
        tree_scroll.pack(side = tk.RIGHT, fill = tk.Y)
        
        # Create a Treeview widget for displaying movie search results
        
        movies_tree = ttk.Treeview(tree_frame, yscrollcommand = tree_scroll.set)
        movies_tree.pack()
        
        tree_scroll.config(command = movies_tree.yview)
        
        movies_tree["columns"] = ("Title", "Genre", "Year", "Movie ID")
        
        movies_tree.column("#0", width = 0, minwidth = 0)
        movies_tree.column("Title", anchor = "w", width = 450)
        movies_tree.column("Genre", anchor = "w", width = 350)
        movies_tree.column("Year", anchor = tk.CENTER, width = 100)
        movies_tree.column("Movie ID", anchor = tk.CENTER, width = 100)
        
        movies_tree.heading("#0", text = "")
        movies_tree.heading("Title", text = "Title", anchor = tk.CENTER)
        movies_tree.heading("Genre", text = "Genre", anchor = tk.CENTER)
        movies_tree.heading("Year", text = "Year", anchor = tk.CENTER)
        movies_tree.heading("Movie ID", text = "Movie ID", anchor = tk.CENTER)
        
        count = 0
        for record in records:
            movies_tree.insert(parent = "", index = "end", iid = count, text = "",
                                    values = (record[0], record[1], record[2], record[3]))
            count += 1
            
        return    
             
        
    # Create function for adding a movie to user history after movie ID has been entered and add button has been clicked    
    def add_movie(self, master):
        movie_ID = self.movie_ID_text_entry.get()
        
        if movie_ID == "":
            messagebox.showerror(title = "Invalid Movie ID", message = "You did not enter a movie ID!")
            return
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        # Check if movie is already in history
        c.execute("SELECT * FROM user_history WHERE username = ? AND movie_ID = ?",
                  (master.username, int(movie_ID)))
        records = c.fetchall()
        if len(records) != 0:
            messagebox.showerror(title = "Invalid Movie ID", 
                                 message = "That movie is already in your history!")
            
            conn.close()
            return
        
        # Check if entered movie ID exists in records
        c.execute("SELECT * FROM movie_info WHERE movie_ID = ?", (int(movie_ID),))
        records = c.fetchall()
        if len(records) == 0:
            messagebox.showerror(title = "Invalid Movie ID",
                                 message = "That movie ID does not exist in our records!")
            
            conn.close()
            return
        
        c.execute("SELECT title FROM movie_info WHERE movie_ID = ?", (int(movie_ID),))
        movie_title = c.fetchall()[0][0]
        
        # Display selected movie in messagebox and ask user if they are sure they want to add it
        decision = messagebox.askquestion(title = "Movie Selected",
                                          message = "You have decided to add '" + movie_title + "' to your history. " +
                                          "Is this correct?")
        if decision == "yes":
            c.execute("INSERT INTO user_history VALUES (:username, :user_ID, :movie_ID)",
                  {
                     "username": master.username,
                     "user_ID": master.user_ID,
                     "movie_ID": int(movie_ID)
                  })
            
            c.execute("SELECT * FROM user_history WHERE username = ?", (master.username,))
            records = c.fetchall()
            conn.commit()
            conn.close()
            
            # Update the movie count
            self.movie_count_label = tk.Label(self, text = "Movies added: {}/20".format(len(records)))
            self.movie_count_label.grid(row = 6, column = 4, columnspan = 2, padx = (70, 0))
            
            # Delete text from ID entry box
            self.movie_ID_text_entry.delete(0, tk.END)
            
        else:
            return
     
    # Create function for finishing the process of user history creation once amount of movies has reached or exceeded 20    
    def finish(self, master):
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        # Check if the user has an existing history and enough movies
        c.execute("SELECT * FROM user_history WHERE username = ?", (master.username,))
        records = c.fetchall()
        
        if len(records) < 20:
            messagebox.showerror(title = "Invalid History Size",
                                 message = "You must add at least 20 movies to your history!")
            conn.close()
        
        # Select movie IDs from user_history
        # Check which movie IDs already exist in train_set for this given user's ID
        # Append new positive samples to train_set
         
        c.execute("SELECT movie_ID FROM user_history WHERE username = ?", (master.username,))    
        all_movie_IDs = set([movie[0] for movie in c.fetchall()])
        c.execute("SELECT movie_ID FROM train_set WHERE user_ID = ?", (master.user_ID,))
        train_set_movie_IDs = set([movie[0] for movie in c.fetchall()])
        movie_IDs = list(all_movie_IDs - train_set_movie_IDs)
        
        for ID in movie_IDs:
            c.execute("INSERT INTO train_set VALUES (:user_ID, :movie_ID, :interaction)",
                         {
                         "user_ID": master.user_ID,
                         "movie_ID": ID,
                         "interaction": 1
                         })    
        
        conn.commit()
        conn.close()
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        # Append negative samples
        for sample in range(1, len(movie_IDs) * 4):
            c.execute("SELECT movie_ID FROM movie_info ORDER BY RANDOM() LIMIT 1")
            rand_movie_ID = c.fetchall()[0][0]
            while rand_movie_ID in movie_IDs:
                c.execute("SELECT movie_ID FROM movie_info ORDER BY RANDOM() LIMIT 1")
                rand_movie_ID = c.fetchall()[0][0]
            
            c.execute("INSERT INTO train_set VALUES (:user_ID, :movie_ID, :interaction)",
                     {
                     "user_ID": master.user_ID,
                     "movie_ID": rand_movie_ID,
                     "interaction": 0
                     })
            
        c.execute("UPDATE user_info SET trained = 0 WHERE username = ?",
                  (master.username,))
        master.trained = 0
        
        conn.commit()
        conn.close()    
        master.change_frame(HistoryPage)
        
    def train_NeuMF(self, master):
        if master.trained == 1:
            messagebox.showerror(title = "Training Error",
                                 message = "The NeuMF network has already been trained with your current history. " +
                                 "To enable training, add more movies to your history and then click 'Finish'.")
            return
            
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        c.execute("SELECT * FROM train_set WHERE user_ID = ?", (master.user_ID,))
        records = c.fetchall()
        if len(records) == 0:
            messagebox.showerror(title = "Training Error",
                                 message = "Your history has not been added to the training set. Return to 'Edit " +
                                 "History' and click 'Finish'.")
            return
            
        if isfile("application data/encodings and saved model/user_ID_encoding.npy"):
            remove("application data/encodings and saved model/user_ID_encoding.npy")
            
        if isfile("application data/encodings and saved model/movie_ID_encoding.npy"):  
            remove("application data/encodings and saved model/movie_ID_encoding.npy")
            
        conn = sql.connect("application data/database.db")
        c = conn.cursor()   
        
        # Select train_set from database and save as numpy array
        c.execute("SELECT user_ID, movie_ID, interaction FROM train_set")
        train_set = np.array(c.fetchall())
        conn.close()
        
        # Encode the user and movie IDs in train_set
        # Save encoding for a given type of ID in a .npy file
        
        user_ID_enc = LabelEncoder()
        user_ID_enc.fit(train_set[:, 0])
        np.save("application data/encodings and saved model/user_ID_encoding.npy", user_ID_enc.classes_)
        train_set[:, 0] = user_ID_enc.transform(train_set[:, 0])
        
        movie_ID_enc = LabelEncoder()
        movie_ID_enc.fit(train_set[:, 1])
        np.save("application data/encodings and saved model/movie_ID_encoding.npy", movie_ID_enc.classes_)
        train_set[:, 1] = movie_ID_enc.transform(train_set[:, 1])
        
        train(model = NeuMF(num_users = np.unique(train_set[:, 0]).shape[0],
                            num_items = np.unique(train_set[:, 1]).shape[0],
                            gmf_embedding_dim = 16, mlp_embedding_dim = 16),
             x_train = [train_set[:, 0], train_set[:, 1]], y_train = train_set[:, 2],
             batch_size = 8192, epochs = 5, lr = 0.0001, save_name = "trained_NeuMF",
             save_model_path = "application data/encodings and saved model")
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        # Update 'trained' value in user_info and master.trained value
        c.execute("UPDATE user_info SET trained = 1 WHERE username = ?",
                  (master.username,))
        master.trained = 1
        conn.commit()
        conn.close()
        
        # Create label widget to inform user that training has finished
        finished_label = tk.Label(self, text = "Training has finished! You can now head to 'Recommend' and see what " +
                                  "we think you will like.", wraplength = round(WINDOW_WIDTH/2))
        finished_label.grid(row = 4, column = 2, columnspan = 3, pady = 15)
        return
        

class RecommendPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)     
        
        # Define buttons for switching between pages after user login or creation
        # Username displayed to the left of buttons
        
        username_display = tk.Label(self, text = master.username, font = master.button_font,)
        username_display.grid(row = 0, column = 0, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        home_button = tk.Button(self, text = "Home", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                command = lambda: master.change_frame(HomePage))
        home_button.grid(row = 0, column = 1, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        bucket_list_button = tk.Button(self, text = "Bucket List", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                       borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                       command = lambda: master.change_frame(BucketListPage))
        bucket_list_button.grid(row = 0, column = 2, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        history_button = tk.Button(self, text = "History", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                   borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                   command = lambda: master.change_frame(HistoryPage))
        history_button.grid(row = 0, column = 3, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
                 
        recommend_button = tk.Button(self, text = "Recommend", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                     borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                     command = lambda: master.change_frame(RecommendPage))
        recommend_button.grid(row = 0, column = 4, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        logout_button = tk.Button(self, text = "Logout", width = MENU_BTN_WIDTH, height = MENU_BTN_HEIGHT,
                                  borderwidth = BTN_BORD_WIDTH, bg = "grey", font = master.button_font,
                                  command = lambda: master.change_frame(LoginPage))
        logout_button.grid(row = 0, column = 5, padx = MENU_BTN_PADX, pady = MENU_BTN_PADY)
        
        # Create info for user
        info_label = tk.Label(self, text = "The 'Recommend Movies' button creates a random selection of 100 movies " +
                              "from our database. The trained NeuMF network then ranks these movies, for your user ID, " +
                              "based on its confidence that you'll watch a given movie. The 10 movies it thinks you are " +
                              "most likely to watch will be displayed below. More detailed information can be displayed " +
                              "for a given movie by entering its ID and clicking 'Movie Info'. You can choose whether or " +
                              "not to add the movie to your bucket list with 'Save Recommendation'.",
                              wraplength = round(WINDOW_WIDTH/2))
        info_label.grid(row = 1, column = 2, columnspan = 3, pady = 15)
        
        # Create dropdown menu for choosing genre
        genre_label = tk.Label(self, text = "Genre")
        genre_label.grid(row = 2, column = 2, columnspan = 2, pady = (15, 5))
        self.genre = tk.StringVar()
        self.genre.set("Any Genre")
        genre_dropdown = tk.OptionMenu(self, self.genre, "Any Genre", "Action", "Adventure", "Animation",
                                       "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                                       "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
                                       "Western")
        genre_dropdown.config(width = 14)
        genre_dropdown.grid(row = 2, column = 3, columnspan = 2, pady = (15, 5))
        
        # Create label and text entry box for movie IDs
        movie_ID_entry_label = tk.Label(self, text = "Movie ID")
        movie_ID_entry_label.grid(row = 3, column = 2, columnspan = 2, pady = (5, 15))
        self.movie_ID_text_entry = tk.Entry(self, width = 20)
        self.movie_ID_text_entry.grid(row = 3, column = 3, columnspan = 2, pady = (5, 15))
        
        
        # Create button for movie recommendations
        recommend_movies_button = tk.Button(self, text = "Recommend Movies", borderwidth = BTN_BORD_WIDTH, width = 20,
                                       bg = "grey", font = master.button_font,
                                       command = lambda: self.recommend(master))
        recommend_movies_button.grid(row = 4, column = 2, pady = 15)
        
        # Create button for viewing more detailed information for a given movie ID
        recommend_movies_button = tk.Button(self, text = "Movie Info", borderwidth = BTN_BORD_WIDTH, width = 20,
                                       bg = "grey", font = master.button_font,
                                       command = lambda: self.movie_info(master))
        recommend_movies_button.grid(row = 4, column = 3, pady = 15)
        
        # Create button for adding a movie recommendation to the bucket list
        save_recommendation_button = tk.Button(self, text = "Save Recommendation",
                                               borderwidth = BTN_BORD_WIDTH, width = 20,
                                               bg = "grey", font = master.button_font,
                                               command = lambda: self.save_recommendation(master))
        save_recommendation_button.grid(row = 4, column = 4, pady = 15)
        
        # Create a frame to pack a Treeview widget into
        tree_frame = tk.Frame(self, width = 1000, height = round(WINDOW_HEIGHT/2))
        tree_frame.grid(row = 5, column = 0, columnspan = 6, pady = 15)
        
        # Create a scrollbar for the frame
        tree_scroll = tk.Scrollbar(tree_frame)
        tree_scroll.pack(side = tk.RIGHT, fill = tk.Y)
        
        # Create a Treeview widget for displaying user records
        
        self.recommended_movies_tree = ttk.Treeview(tree_frame, yscrollcommand = tree_scroll.set)
        self.recommended_movies_tree.pack()
        
        tree_scroll.config(command = self.recommended_movies_tree.yview)
        
        self.recommended_movies_tree["columns"] = ("Title", "Genre", "Year", "Movie ID")
        
        self.recommended_movies_tree.column("#0", width = 0, minwidth = 0)
        self.recommended_movies_tree.column("Title", anchor = "w", width = 450)
        self.recommended_movies_tree.column("Genre", anchor = "w", width = 350)
        self.recommended_movies_tree.column("Year", anchor = tk.CENTER, width = 100)
        self.recommended_movies_tree.column("Movie ID", anchor = tk.CENTER, width = 100)
        
        self.recommended_movies_tree.heading("#0", text = "")
        self.recommended_movies_tree.heading("Title", text = "Title", anchor = tk.CENTER)
        self.recommended_movies_tree.heading("Genre", text = "Genre", anchor = tk.CENTER)
        self.recommended_movies_tree.heading("Year", text = "Year", anchor = tk.CENTER)
        self.recommended_movies_tree.heading("Movie ID", text = "Movie ID", anchor = tk.CENTER)
    
    # Create function for generating random movies and choosing 10 with highest interaction confidence
    def recommend(self, master):
        if master.trained == 0:
            messagebox.showerror(title = "Training Error",
                                 message = "You have not trained the NeuMF network on your history yet!")
            return
        
        # Delete any current rows in treeview widget
        self.recommended_movies_tree.delete(*self.recommended_movies_tree.get_children())
        
        # Store the train_set movie IDs for the current user in a list
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        if self.genre.get() == "Any Genre":
            c.execute("""SELECT movie_ID FROM movie_info
                      WHERE movie_ID NOT IN (SELECT movie_ID FROM train_set WHERE user_ID = ?)
                      ORDER BY RANDOM() LIMIT 100""", (master.user_ID,))
        else:
            c.execute("""SELECT movie_ID FROM movie_info
                      WHERE movie_ID NOT IN (SELECT movie_ID FROM train_set WHERE user_ID = ?)
                      AND genre LIKE ?
                      ORDER BY RANDOM() LIMIT 100""", (master.user_ID, "%" + self.genre.get() + "%"))
                     
        movie_IDs_100 = [record[0] for record in c.fetchall()]  
        conn.close()
           
        # Encode user_ID and movie_IDs with the scheme used for training
        user_ID_enc = LabelEncoder()
        user_ID_enc.classes_ = np.load('application data/encodings and saved model/user_ID_encoding.npy')
        user_ID_100 = user_ID_enc.transform(np.array([master.user_ID] * 100))
        movie_ID_enc = LabelEncoder()
        movie_ID_enc.classes_ = np.load('application data/encodings and saved model/movie_ID_encoding.npy')
        movie_IDs_100 = movie_ID_enc.transform(np.array(movie_IDs_100))
           
        # Load trained NeuMF network and make predictions for interactions with movies
        trained_NeuMF = load_model("application data/encodings and saved model/trained_NeuMF.h5")
        preds = np.squeeze(trained_NeuMF.predict([user_ID_100, movie_IDs_100]),
                               axis = 1)  
        
        # Sort movies by confidence of interaction and then inverse transform to get original movie IDs
        top10_movie_IDs = [movie_IDs_100[i] for i in np.argsort(preds)[::-1][0:10].tolist()]
        top10_movie_IDs = movie_ID_enc.inverse_transform(np.array(top10_movie_IDs))
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        # Use movie_info table to find information about the top 10 movies
        # Insert rows into treeview widget
        count = 0
        for ID in top10_movie_IDs:
            c.execute("SELECT title, genre, year, movie_ID FROM movie_info WHERE movie_ID = ?", (int(ID),))
            movie = c.fetchall()[0]
            self.recommended_movies_tree.insert(parent = "", index = "end", iid = count, text = "",
                                    values = (movie[0], movie[1], movie[2], movie[3]))
            count += 1
        
        conn.close()
        return
    
    def movie_info(self, master):
        movie_ID = self.movie_ID_text_entry.get()
        
        if movie_ID == "":
            messagebox.showerror(title = "Invalid Movie ID", message = "You did not enter a movie ID!")
            return
        
        movie_info_window = tk.Toplevel(master)
        movie_info_window.title("Movie Info")
        movie_info_window.geometry(str(round(WINDOW_WIDTH * 7/12)) + "x" + str(round(WINDOW_HEIGHT * 4/3)))
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        c.execute("SELECT IMDb_ID, genre FROM movie_info WHERE movie_ID = ?", (movie_ID,))
        record = c.fetchall()
        IMDb_ID = str(record[0][0])
        genres_string = str(record[0][1])
        
        while len(IMDb_ID) < 7:
            IMDb_ID = "0" + IMDb_ID
        
        IMDb_ID = "tt" + IMDb_ID    
        
        url = "https://imdb8.p.rapidapi.com/title/get-overview-details"

        querystring = {"tconst":IMDb_ID,"currentCountry":"GB"}
        
        headers = {
            'x-rapidapi-key': RAPID_API_KEY,
            'x-rapidapi-host': "imdb8.p.rapidapi.com"
            }
        
        response = requests.get(url, headers=headers, params=querystring)
        
        movie_info = response.json()
        
        image_url = movie_info["title"]["image"]["url"]
        
        self.movie_poster = Image.open(requests.get(image_url, stream=True).raw).resize((340, 503))
        self.movie_poster = ImageTk.PhotoImage(self.movie_poster)
        
        image_label = tk.Label(movie_info_window, image = self.movie_poster)
        image_label.grid(row = 0, column = 0, columnspan = 2, pady = (10, 15))
        
        X_PADDING_L = (45, 40)
        X_PADDING_R = (40, 25)
        
        title_label = tk.Label(movie_info_window, text = "Title", font = master.button_font)
        title_label.grid(row = 1, column = 0, padx = X_PADDING_L, pady = (15, 10))
        title = tk.Label(movie_info_window, text = movie_info["title"]["title"])
        title.grid(row = 1, column = 1, padx = X_PADDING_R, pady = (15, 10))
        
        genres_label = tk.Label(movie_info_window, text = "Genre", font = master.button_font)
        genres_label.grid(row = 2, column = 0, padx = X_PADDING_L, pady = 10)
        genres = tk.Label(movie_info_window, text = genres_string, wraplength = round(WINDOW_WIDTH/4))
        genres.grid(row = 2, column = 1, padx = X_PADDING_R, pady = 10)
        
        year_label = tk.Label(movie_info_window, text = "Year", font = master.button_font)
        year_label.grid(row = 3, column = 0, padx = X_PADDING_L, pady = 10)
        year = tk.Label(movie_info_window, text = movie_info["title"]["year"])
        year.grid(row = 3, column = 1, padx = X_PADDING_R, pady = 10)
        
        runtime_label = tk.Label(movie_info_window, text = "Runtime (mins)", font = master.button_font)
        runtime_label.grid(row = 4, column = 0, padx = X_PADDING_L, pady = 10)
        runtime = tk.Label(movie_info_window, text = movie_info["title"]["runningTimeInMinutes"])
        runtime.grid(row = 4, column = 1, padx = X_PADDING_R, pady = 10)
        
        rating_label = tk.Label(movie_info_window, text = "IMDb Rating", font = master.button_font)
        rating_label.grid(row = 5, column = 0, padx = X_PADDING_L, pady = 10)
        rating = tk.Label(movie_info_window, text = "{}/10 ({})".format(movie_info["ratings"]["rating"],
                       movie_info["ratings"]["ratingCount"]))
        rating.grid(row = 5, column = 1, padx = X_PADDING_R, pady = 10)
        
        plot_label = tk.Label(movie_info_window, text = "Plot Summary", font = master.button_font)
        plot_label.grid(row = 6, column = 0, padx = X_PADDING_L, pady = 15)
        plot_frame = tk.Frame(movie_info_window, width = round(WINDOW_WIDTH/3),
                              height = round(WINDOW_HEIGHT * 4/27))
        plot_frame.grid(row = 6, column = 1, padx = X_PADDING_R, pady = 15)
        plot_frame.pack_propagate(0) # Stops child widgets of label_frame from resizing it
        plot = tk.Label(plot_frame, text = movie_info["plotOutline"]["text"],
                        wraplength = round(WINDOW_WIDTH/3))
        plot.pack(expand=True, fill='both')
        
        movie_info_window.grid_columnconfigure(0, weight = 1)
        movie_info_window.grid_columnconfigure(1, weight = 1)
        
    
    def save_recommendation(self, master):
        if master.trained == 0:
            messagebox.showerror(title = "Training Error",
                                 message = "You have not trained the NeuMF network on your history yet!")
            return
        
        movie_ID = self.movie_ID_text_entry.get()
        
        if movie_ID == "":
            messagebox.showerror(title = "Invalid Movie ID", message = "You did not enter a movie ID!")
            return
        
        conn = sql.connect("application data/database.db")
        c = conn.cursor()
        
        # Check if movie is already in user_bucket_list for this user
        c.execute("SELECT * FROM user_bucket_list WHERE username = ? AND movie_ID = ?",
                  (master.username, int(movie_ID)))
        records = c.fetchall()
        if len(records) != 0:
            messagebox.showerror(title = "Invalid Movie ID", 
                                 message = "That movie is already in your bucket list!")
            
            conn.close()
            return
        
        # Check if entered movie ID exists in records
        c.execute("SELECT * FROM movie_info WHERE movie_ID = ?", (int(movie_ID),))
        records = c.fetchall()
        if len(records) == 0:
            messagebox.showerror(title = "Invalid Movie ID",
                                 message = "That movie ID does not exist in our records!")
            
            conn.close()
            return
        
        c.execute("SELECT title FROM movie_info WHERE movie_ID = ?", (int(movie_ID),))
        movie_title = c.fetchall()[0][0]
        
        # Display selected movie in messagebox and ask user if they are sure they want to add it
        decision = messagebox.askquestion(title = "Movie Selected",
                                          message = "You have decided to add '" + movie_title + "' to your bucket " +
                                          "list. Is this correct?")
        if decision == "yes":
            c.execute("INSERT INTO user_bucket_list VALUES (:username, :user_ID, :movie_ID)",
                  {
                     "username": master.username,
                     "user_ID": master.user_ID,
                     "movie_ID": int(movie_ID)
                  })
            
            conn.commit()
            conn.close()
            
            # Delete text from ID entry box
            self.movie_ID_text_entry.delete(0, tk.END)
            
        else:
            return
        
        
           
if __name__ == "__main__":
    app = App()
    app.mainloop()        
        
        
        
        
            

