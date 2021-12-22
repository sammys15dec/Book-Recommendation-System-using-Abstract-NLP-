import streamlit as st
import pickle
import re
pip install sklearn
from sklearn import preprocessing

st.title('Book Recommendation System')
st.subheader("Natural Language Processing")


model1 = pickle.load(open('model1.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
model2 = pickle.load(open('model2-transform.pkl','rb'))

def recommend(category):
    recommended_books = model1[model1['category'] == category].sort_values(
        by='Books', ascending=False)
    rec_book_list = recommended_books[['Books']]
    rec_book_list = rec_book_list.reset_index(drop=True)
    return rec_book_list

#text cleaning,lowercase,removing special characters
def clean_summary(text):
    # removing everything other than alphabets and numbers with spaces
    text = re.sub('\W+', ' ', text)
    text = text.lower()  # converts all the text to lowercase
    return text



text_in = st.text_area('Enter Abstract/Content')
rec = [clean_summary(text_in)]

if st.button('Predict'):
    t = model2.transform(rec)
    le = preprocessing.LabelEncoder()
    le.fit_transform(model1.category)
    pr = le.inverse_transform(model.predict(t))
    
    
    st.write('Books based on the prediction are :',recommend(pr[0]))
