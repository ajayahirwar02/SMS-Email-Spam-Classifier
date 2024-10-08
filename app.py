import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

#set page configuration
st.set_page_config(
    page_title="SMS/Email Spam Classifier" 
)


nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))

model = pickle.load(open('model.pkl','rb'))

st.title("SMS/Email Spam Classifier")

input_sms = st.text_area("Enter the SMS/Email")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

#design footer
st.markdown(
    """
    <style>
    .footer { 
        #link{
        text-decoration: none;
        }
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 14px;d
        box-shadow: 0px -1px 5px rgba(0, 0, 0, 0.1);
    }
    </style>
    <div class="footer">
        <p> <a id ="link" href = "https://www.linkedin.com/in/ajayahirwar02/">Developed by Ajay Ahirwar</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
