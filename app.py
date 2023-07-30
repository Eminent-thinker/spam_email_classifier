import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download("punkt")
nltk.download("stopwords")

ps = PorterStemmer()

def transform_message(message):
    message = message.lower()
    words = nltk.word_tokenize(message)
    transformed_words =[]
    for x in words:
        if x not in stopwords.words('english') and x.isalnum():
            root_words = ps.stem(x)
            transformed_words.append(root_words)
    return ' '.join(transformed_words)

cv = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')
st.title('SMS/EMAIL SPAM CLASSIFIER')
message_input = st.text_area('Enter your message here')
transformed_text = transform_message(message_input)
if st.button('Predict'):
   vector_input = cv.transform([transformed_text])
   result = model.predict(vector_input)[0]

   if result == 1:
    st.success('Not Spam')
   else:
    st.error('Spam')

