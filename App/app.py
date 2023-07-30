import streamlit as st
import pickle

def transform_message(message):
    message = message.lower()
    words = nltk.word_tokenize(message)
    transformed_words =[]
    for x in words:
        if x not in stopwords.words('english') and x.isalnum():
            root_words = ps.stem(x)
            transformed_words.append(root_words)
    return ' '.join(transformed_words)

cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
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

