import sys
import os
import re
import torch
import base64
import pandas as pd 
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.getcwd())

st.title("ArXiV Paper Recommender")

def set_background(main_bg):
    main_bg_ext = "jpg"
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_background("../images/p1.jpg")

topic = st.text_input('What kind of paper would you wish to be recommended?', 'I want to read a paper on Bayesian Optimization!')
number = st.number_input('Show me these many papers.', min_value=1, max_value=10, value=3, step=1)

def process_text(text):
    rep = {"\n": " ", "(": "", ")": "", "!": ""}
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text).lower()
    return text

def get_cosine_similarity(feature_vec_1, feature_vec_2):    
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]

def get_model():
	device = 'cuda' if torch.cuda.is_available() else None
	model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
	return model

if st.button("GO!"):
	prompt = process_text(topic)
	model = get_model()
	prompt_embedded = model.encode(prompt)
	df_embed = pd.read_pickle('../data/embeddings_pkl.pkl').drop_duplicates(subset=['titles'])
	df_embed["similarity_scores"] = df_embed["abstracts_embeddings"].apply(lambda x: get_cosine_similarity(x, prompt_embedded))
	top_n = df_embed.nlargest(number, 'similarity_scores').head(5)["titles"].to_list()
	st.text(" ")
	st.subheader('Have a look at the following: :sunglasses:')
	for rec_title in top_n:
		st.markdown("-> " + rec_title)