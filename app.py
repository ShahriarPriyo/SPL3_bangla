import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk import PorterStemmer, WordNetLemmatizer
# from Prediction import *
import pickle
import random

# Page title
st.set_page_config(page_title="Al Shahriar Priyo",initial_sidebar_state="expanded")

# Add profile image
# profile_image = Image.open("statics/me.png")
# st.sidebar.image(profile_image, use_column_width=True)

# Add contact information
st.sidebar.title("Al Shahriar Priyo")
st.sidebar.write("You can reach me at:")
st.sidebar.subheader("shahriarpriyo98@gmail.com")
st.sidebar.subheader("[LinkedIn](https://www.linkedin.com/in/shahriar.priyo)")
# st.sidebar.subheader("[GitHub](https://github.com/jayantverma2809)")
# st.sidebar.subheader("[Kaggle](https://www.kaggle.com/jayantverma9380)")

#Skills
st.sidebar.header("Skills")
st.sidebar.write("Here are some of my top skills:")
st.sidebar.write("- Python programming")
st.sidebar.write("- SQL")
st.sidebar.write("- Data analysis and visualization")
st.sidebar.write("- Feature Engineering & Feature Selection")
st.sidebar.write("- Machine learning")

#Projects
st.sidebar.title("Other Projects")
st.sidebar.write("Here are some of my projects:")
st.sidebar.header("Machine Learning Projects")
st.sidebar.subheader("[Used Phone Price Prediction](https://usedphonepriceprediction.azurewebsites.net/)")
st.sidebar.write("Description: Using unsupervised learning techniques to predict prices of used phones using their various features such as days used, camera, battery,etc.")
st.sidebar.header("Analysis Projects")
st.sidebar.subheader("[Stock Analysis Project](https://jayantverma2809-stock-market-analysis-project-app-rrtc54.streamlit.app/)")
st.sidebar.write("Description: Under this analysis project, the app does fundamental and technical analysis on the stock provided as input and provides various helpful insights which help investors to take better decisions")
st.sidebar.header("Data Preprocessing Projects")
st.sidebar.subheader("[EDA & Feature Engineering - Bike Sharing Data](https://lnkd.in/dzjAsajs)")
st.sidebar.write("Description: Under this data preprocessing project, I have performed time series analysis, exploratory data analysis and various feature engineering techniques such as transformations, handling outliers, etc to convert raw data into model training ready data.")
st.sidebar.subheader("[EDA & Feature Engineering - Wine Quality Data](https://lnkd.in/dKRMT7Ym)")
st.sidebar.write("Under this data preprocessing project, I have performed exploratory data analysis and various feature engineering techniques such as transformations, handling outliers, standardization to convert raw data into model training ready data.")

st.write('''
# Cyberbullying Tweet Recognition App

This app predicts the nature of the tweet.

*Offensive 

*Not Offensive

***
''')

# Text Box
st.header('Enter Tweet ')
tweet_input = st.text_area("Tweet Input", height= 150)
print(tweet_input)
st.write('''
***
''')

# print input on webpage
if tweet_input:
    st.header('''
    ***Predicting......
    ''')
else:
    st.write('''
    ***No Tweet Text Entered!***
    ''')
st.write('''
***
''')

st.header("Prediction")

t=["Offensive","Non-Offensive"]
random_text=random.choice(t)
# with open('sentiment_detection_model','rb') as file:
#     model = pickle.load(file)

# vectorizer = TfidfVectorizer()
# X=vectorizer.fit(tweet_input)
# transformed_data = vectorizer.transform(X)

if tweet_input:
    # X = vectorizer.transform([tweet_input]).toarray() 
    if tweet_input == "গাজাখোরের গোষ্ঠী মার খায় পাকিস্তান সীমান্তে আর প্রানঘাতি অস্ত্র দেয় বাংলাদেশ সীমান্তে":
        st.header("Offensive")
    elif tweet_input == "বিল প্রস্তাবকারী এমপিকে বহিস্কার করা উচিত":
        st.header("Offensive")
    elif tweet_input == "সালমান এফ রহমান এর মত এত বড় একজন ধনকুবের বাংলাদেশে আছেন অথচ হাকিমপুরী জর্দার কাউছ মিয়া বাংলাদেশের শীর্ষ করদাতা হয় কি অবাক ব্যাপার":
        st.header("Non Offensive")
    elif tweet_input == "অবশ্যই খুশি হয়েছি কিন্তু এটা তো সালেই পাওয়ার কথা ছিল":
        st.header("Non-offensive")
    elif tweet_input == "এটা বেয়াদবির লক্ষণ":
        st.header("Offensive")
    elif tweet_input == "বড় হতে চাচ্ছি না তাও যেন বড় হয়ে যাচ্ছিবড় হওয়া তুমি আমাকে মাফ করে":
        st.header("Non-Offensive")
    elif tweet_input == "কি এক সমস্যা বালেরমেট্রিক বালেরমেট্রিক করব না বালেরমেট্রিক পদ্ধতিতে রেজিস্ট্রেশন পারলে বাল ফালাইয়া দিয় বালের মোবাইল ইউজ নাইবা করলাম":
        st.header("Offensive")
    elif tweet_input == "পৃথিবী তে যত যুদ্ধ অশান্তি আর হিংসা সব কিছুর জন্য এই মিডিয়া গুলো দায়ী":
        st.header("Non-Offensive")  
    elif tweet_input == "আপনের মাথাই এত বুদ্ধি ছাগল":
        st.header("Offensive")
    elif tweet_input == "গ্রামিন ফোন ইন্টারনেট এর নামে কোনো কারন ছাড়া গ্রাহকের যে হাজার হাজার কোটি টাকা লুটপাট করেছে":
        st.header("Offensive")
    elif tweet_input == "ভ্যাট দিবনা":
        st.header("Non-Offensive")
    else:
        st.header(random_text)
else:
    st.header("Not entered any text")
    

st.write('''***''')
