import streamlit as st
import openai
import pandas as pd
import numpy as np
import time


from datetime import datetime
import os
def log_results(dataset_name, method, num_topics, coherence_umass, coherence_cv, topic_stability, topic_diversity, topic_uniqueness, processing_time):
    log_file = "topic_modeling_logs.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = {
        "Date": [timestamp],
        "Dataset": [dataset_name],
        "Method": [method],
        "Topics": [num_topics],
        "UMass": [coherence_umass],
        "C_V": [coherence_cv],
        "Stability": [topic_stability],
        "Diversity": [topic_diversity],
        "Uniqueness": [topic_uniqueness],
        "Processing Time (s)": [round(processing_time, 2)]
    }
    log_df = pd.DataFrame(log_data)
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)

def log_topic_details(dataset_name, method, topic_num, top_words, llm_label=None, processing_time=None):
    detail_file = "topic_modeling_details.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "Date": [timestamp],
        "Dataset": [dataset_name],
        "Method": [method],
        "Topic #": [topic_num],
        "Top Words": [", ".join(top_words)],
        "LLM Label": [llm_label if llm_label else ""],
        "Processing Time (s)": [round(processing_time, 2) if processing_time is not None else ""]
    }
    log_df = pd.DataFrame(log_entry)
    if os.path.exists(detail_file):
        log_df.to_csv(detail_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(detail_file, index=False)

from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from bertopic import BERTopic
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from top2vec import Top2Vec

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import pyLDAvis.gensim_models
import pyLDAvis


from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun

def lemmatize_text(text):
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)
    return " ".join([
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in tagged
        if word.isalpha() and word not in ENGLISH_STOP_WORDS
    ])



def get_openai_embeddings(docs, api_key):
    import openai
    client = openai.OpenAI(api_key=api_key)

    embeddings = []
    for doc in docs:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=doc
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            embeddings.append([0.0] * 1536)
            print(f"Embedding failed: {e}")
    return embeddings

def preprocess_text(texts):
    return [" ".join(str(t).lower().split()) for t in texts if isinstance(t, str)]

def train_lda(docs, num_topics=10):
    docs = [doc for doc in docs if doc.strip()]
    if not docs:
        raise ValueError("All documents are empty after preprocessing or contain only stop words.")

    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(docs)
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(doc_term_matrix)
    return lda_model, vectorizer

def train_bertopic(docs):
    cleaned_docs = [lemmatize_text(doc) for doc in docs]
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(cleaned_docs)
    return topic_model

def train_bertopic_llm(docs, api_key):
    cleaned_docs = [lemmatize_text(doc) for doc in docs]
    topic_model = BERTopic(embedding_model=lambda d: get_openai_embeddings(d, api_key))
    topics, probs = topic_model.fit_transform(cleaned_docs)
    return topic_model

def train_top2vec(docs):
    lemmatized_docs = [lemmatize_text(doc) for doc in docs]
    model = Top2Vec(documents=lemmatized_docs, speed="learn", workers=4)
    return model

st.title("Unified Topic Modeling Application")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    st.success("Dataset uploaded successfully!")
    
    text_column = st.selectbox("Select the column containing text data", df.columns)
    additional_column = st.selectbox("Optionally, select an additional column to combine", ["None"] + list(df.columns))
    if additional_column != "None" and additional_column != text_column:
        df["combined_text"] = df[text_column].astype(str) + " " + df[additional_column].astype(str)
        docs = preprocess_text(df["combined_text"])
    else:
        docs = preprocess_text(df[text_column])
    
    method = st.selectbox("Choose Topic Modeling Method", ["LDA", "BERTopic", "BERTopic (LLM)", "Top2Vec"])
    
    generate_labels = st.checkbox("Generate LLM-based topic labels")
    api_key = None
    if generate_labels:
        api_key = st.text_input("Enter your OpenAI API Key", type="password")
    
    if method == "LDA":
        num_topics = st.number_input("Number of Topics", min_value=2, max_value=100, value=10, step=1)
        num_words = st.number_input("Words per Topic", min_value=3, max_value=30, value=10, step=1)

    if st.button("Run Topic Modeling"):
        with st.spinner("Processing..."):
            pass
            
            if method == "LDA":
                start_time = time.time()
                lda_model, vectorizer = train_lda(docs, int(num_topics))
                st.success(f"LDA Model with {int(num_topics)} Topics Trained!")

                topic_words = []
                for idx, topic in enumerate(lda_model.components_):
                    terms = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-int(num_words)-1:-1]]
                    topic_words.append(terms)
           
                processing_time = time.time() - start_time
                for idx, terms in enumerate(topic_words):
                    st.write(f"**Topic {idx+1}:**", terms)
                   
                    log_topic_details(
                        dataset_name=uploaded_file.name,
                        method=method,
                        topic_num=idx + 1,
                        top_words=terms,
                        processing_time=processing_time
                    )

                from collections import Counter
                tokenized_docs = [doc.lower().split() for doc in docs if isinstance(doc, str)]
                dictionary = Dictionary(tokenized_docs)
                corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
                coherence_model_umass = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='u_mass')
                coherence_model_cv = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
                coherence_umass = coherence_model_umass.get_coherence()
                coherence_cv = coherence_model_cv.get_coherence()
                all_words = [w for t in topic_words for w in t]
                unique_words = set(all_words)
                topic_diversity = len(unique_words) / len(all_words) if all_words else 0
                topic_uniqueness = sum(1 for count in Counter(all_words).values() if count == 1) / len(all_words) if all_words else 0
                total_words = sum(len(t) for t in topic_words)
                repeated = sum(count - 1 for word, count in Counter(all_words).items() if count > 1)
                topic_stability = 1 - (repeated / total_words) if total_words else 0
                st.write(f"**UMass Coherence:** {coherence_umass:.4f}")
                st.write(f"**C_V Coherence:** {coherence_cv:.4f}")
                st.write(f"**Topic Stability:** {topic_stability:.4f}")
                st.write(f"**Topic Diversity:** {topic_diversity:.4f}")
                st.write(f"**Topic Uniqueness:** {topic_uniqueness:.4f}")

              
                st.write(f"⏱️ **Processing Time:** {processing_time:.2f} seconds")
                log_results(
                    dataset_name=uploaded_file.name,
                    method=method,
                    num_topics=len(topic_words),
                    coherence_umass=coherence_umass,
                    coherence_cv=coherence_cv,
                    topic_stability=topic_stability,
                    topic_diversity=topic_diversity,
                    topic_uniqueness=topic_uniqueness,
                    processing_time=processing_time
                )

                if generate_labels and api_key:
                    client = openai.OpenAI(api_key=api_key)

                    def generate_label(words_list):
                        prompt = f"Given the following words: {', '.join(words_list)}, generate a short, descriptive label for the topic."
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": prompt}]
                            )
                            return response.choices[0].message.content.strip()
                        except openai.RateLimitError:
                            st.error("Rate limit exceeded or insufficient quota. Please check your OpenAI billing/plan.")
                            return "LLM label unavailable"
                        except Exception as e:
                            st.error(f"OpenAI API error: {e}")
                            return "LLM label unavailable"

                    st.subheader("LLM-Generated Topic Labels:")
                    for idx, words in enumerate(topic_words):
                        label = generate_label(words)
                        st.write(f"**Topic {idx+1}:** {label}")
                        
                        log_topic_details(
                            dataset_name=uploaded_file.name,
                            method=method,
                            topic_num=idx + 1,
                            top_words=words,
                            llm_label=label,
                            processing_time=processing_time
                        )

            elif method == "BERTopic":
                start_time = time.time()
                topic_model = train_bertopic(docs)
                st.success("BERTopic Model Trained!")
                topics = topic_model.get_topic_info()
                topics = topics[topics["Topic"] != -1] 
                st.subheader("Discovered Topics:")
                topic_words = []
                for _, row in topics.iterrows():
                    topic_id = int(row["Topic"])
                    top_words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
                    topic_words.append(top_words)
                
                processing_time = time.time() - start_time
                for _, row in topics.iterrows():
                    topic_id = int(row["Topic"])
                    top_words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
                    st.write(f"**Topic {topic_id + 1}:** {', '.join(top_words)}")
                    
                    log_topic_details(
                        dataset_name=uploaded_file.name,
                        method=method,
                        topic_num=topic_id + 1,
                        top_words=top_words,
                        processing_time=processing_time
                    )

                from collections import Counter
                tokenized_docs = [doc.lower().split() for doc in docs if isinstance(doc, str)]
                dictionary = Dictionary(tokenized_docs)
                corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
                coherence_model_umass = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='u_mass')
                coherence_model_cv = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
                coherence_umass = coherence_model_umass.get_coherence()
                coherence_cv = coherence_model_cv.get_coherence()
                all_words = [w for t in topic_words for w in t]
                unique_words = set(all_words)
                topic_diversity = len(unique_words) / len(all_words) if all_words else 0
                topic_uniqueness = sum(1 for count in Counter(all_words).values() if count == 1) / len(all_words) if all_words else 0
                total_words = sum(len(t) for t in topic_words)
                repeated = sum(count - 1 for word, count in Counter(all_words).items() if count > 1)
                topic_stability = 1 - (repeated / total_words) if total_words else 0
                st.write(f"**UMass Coherence:** {coherence_umass:.4f}")
                st.write(f"**C_V Coherence:** {coherence_cv:.4f}")
                st.write(f"**Topic Stability:** {topic_stability:.4f}")
                st.write(f"**Topic Diversity:** {topic_diversity:.4f}")
                st.write(f"**Topic Uniqueness:** {topic_uniqueness:.4f}")

                
                st.write(f"⏱️ **Processing Time:** {processing_time:.2f} seconds")
                log_results(
                    dataset_name=uploaded_file.name,
                    method=method,
                    num_topics=len(topic_words),
                    coherence_umass=coherence_umass,
                    coherence_cv=coherence_cv,
                    topic_stability=topic_stability,
                    topic_diversity=topic_diversity,
                    topic_uniqueness=topic_uniqueness,
                    processing_time=processing_time
                )

                if generate_labels and api_key:
                    client = openai.OpenAI(api_key=api_key)

                    def generate_label(words_list):
                        prompt = f"Given the following words: {', '.join(words_list)}, generate a short, descriptive label for the topic."
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": prompt}]
                            )
                            return response.choices[0].message.content.strip()
                        except openai.RateLimitError:
                            st.error("Rate limit exceeded or insufficient quota. Please check your OpenAI billing/plan.")
                            return "LLM label unavailable"
                        except Exception as e:
                            st.error(f"OpenAI API error: {e}")
                            return "LLM label unavailable"

                    st.subheader("LLM-Generated Topic Labels:")
                    for idx, words in enumerate(topic_words):
                        label = generate_label(words)
                        st.write(f"**Topic {idx+1}:** {label}")
                        
                        log_topic_details(
                            dataset_name=uploaded_file.name,
                            method=method,
                            topic_num=idx + 1,
                            top_words=words,
                            llm_label=label,
                            processing_time=processing_time
                        )

            elif method == "BERTopic (LLM)":
                start_time = time.time()
                if not api_key:
                    st.warning("Please provide your OpenAI API key to run BERTopic (LLM).")
                else:
                    topic_model = train_bertopic_llm(docs, api_key)
                    st.success("BERTopic (LLM) Model Trained!")
                    topics = topic_model.get_topic_info()
                    topics = topics[topics["Topic"] != -1]
                    st.subheader("Discovered Topics:")
                    topic_words = []
                    for _, row in topics.iterrows():
                        topic_id = int(row["Topic"])
                        top_words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
                        topic_words.append(top_words)
                    
                    processing_time = time.time() - start_time
                    for _, row in topics.iterrows():
                        topic_id = int(row["Topic"])
                        top_words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
                        st.write(f"**Topic {topic_id + 1}:** {', '.join(top_words)}")
                        
                        log_topic_details(
                            dataset_name=uploaded_file.name,
                            method=method,
                            topic_num=topic_id + 1,
                            top_words=top_words,
                            processing_time=processing_time
                        )

                    from collections import Counter
                    tokenized_docs = [doc.lower().split() for doc in docs if isinstance(doc, str)]
                    dictionary = Dictionary(tokenized_docs)
                    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
                    coherence_model_umass = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='u_mass')
                    coherence_model_cv = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
                    coherence_umass = coherence_model_umass.get_coherence()
                    coherence_cv = coherence_model_cv.get_coherence()
                    all_words = [w for t in topic_words for w in t]
                    unique_words = set(all_words)
                    topic_diversity = len(unique_words) / len(all_words) if all_words else 0
                    topic_uniqueness = sum(1 for count in Counter(all_words).values() if count == 1) / len(all_words) if all_words else 0
                    total_words = sum(len(t) for t in topic_words)
                    repeated = sum(count - 1 for word, count in Counter(all_words).items() if count > 1)
                    topic_stability = 1 - (repeated / total_words) if total_words else 0
                    st.write(f"**UMass Coherence:** {coherence_umass:.4f}")
                    st.write(f"**C_V Coherence:** {coherence_cv:.4f}")
                    st.write(f"**Topic Stability:** {topic_stability:.4f}")
                    st.write(f"**Topic Diversity:** {topic_diversity:.4f}")
                    st.write(f"**Topic Uniqueness:** {topic_uniqueness:.4f}")

                    # Log results to CSV
                    st.write(f"⏱️ **Processing Time:** {processing_time:.2f} seconds")
                    log_results(
                        dataset_name=uploaded_file.name,
                        method=method,
                        num_topics=len(topic_words),
                        coherence_umass=coherence_umass,
                        coherence_cv=coherence_cv,
                        topic_stability=topic_stability,
                        topic_diversity=topic_diversity,
                        topic_uniqueness=topic_uniqueness,
                        processing_time=processing_time
                    )

                    if generate_labels:
                        client = openai.OpenAI(api_key=api_key)

                        def generate_label(words_list):
                            prompt = f"Given the following words: {', '.join(words_list)}, generate a short, descriptive label for the topic."
                            try:
                                response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[{"role": "user", "content": prompt}]
                                )
                                return response.choices[0].message.content.strip()
                            except openai.RateLimitError:
                                st.error("Rate limit exceeded or insufficient quota. Please check your OpenAI billing/plan.")
                                return "LLM label unavailable"
                            except Exception as e:
                                st.error(f"OpenAI API error: {e}")
                                return "LLM label unavailable"

                        st.subheader("LLM-Generated Topic Labels:")
                        for idx, words in enumerate(topic_words):
                            label = generate_label(words)
                            st.write(f"**Topic {idx+1}:** {label}")
                            
                            log_topic_details(
                                dataset_name=uploaded_file.name,
                                method=method,
                                topic_num=idx + 1,
                                top_words=words,
                                llm_label=label,
                                processing_time=processing_time
                            )

            elif method == "Top2Vec":
                start_time = time.time()
                model = train_top2vec(docs)
                st.success("Top2Vec Model Trained!")
                topic_words, word_scores, topic_nums = model.get_topics()
                
                processing_time = time.time() - start_time
                for idx, words in enumerate(topic_words):
                    st.write(f"**Topic {idx+1}:**", words)
                    
                    log_topic_details(
                        dataset_name=uploaded_file.name,
                        method=method,
                        topic_num=idx + 1,
                        top_words=words,
                        processing_time=processing_time
                    )

                from collections import Counter
                tokenized_docs = [doc.lower().split() for doc in docs if isinstance(doc, str)]
                dictionary = Dictionary(tokenized_docs)
                corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
                coherence_model_umass = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='u_mass')
                coherence_model_cv = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
                coherence_umass = coherence_model_umass.get_coherence()
                coherence_cv = coherence_model_cv.get_coherence()
                all_words = [w for t in topic_words for w in t]
                unique_words = set(all_words)
                topic_diversity = len(unique_words) / len(all_words) if all_words else 0
                topic_uniqueness = sum(1 for count in Counter(all_words).values() if count == 1) / len(all_words) if all_words else 0
                total_words = sum(len(t) for t in topic_words)
                repeated = sum(count - 1 for word, count in Counter(all_words).items() if count > 1)
                topic_stability = 1 - (repeated / total_words) if total_words else 0
                st.write(f"**UMass Coherence:** {coherence_umass:.4f}")
                st.write(f"**C_V Coherence:** {coherence_cv:.4f}")
                st.write(f"**Topic Stability:** {topic_stability:.4f}")
                st.write(f"**Topic Diversity:** {topic_diversity:.4f}")
                st.write(f"**Topic Uniqueness:** {topic_uniqueness:.4f}")

                
                st.write(f"⏱️ **Processing Time:** {processing_time:.2f} seconds")
                log_results(
                    dataset_name=uploaded_file.name,
                    method=method,
                    num_topics=len(topic_words),
                    coherence_umass=coherence_umass,
                    coherence_cv=coherence_cv,
                    topic_stability=topic_stability,
                    topic_diversity=topic_diversity,
                    topic_uniqueness=topic_uniqueness,
                    processing_time=processing_time
                )

                if generate_labels and api_key:
                    client = openai.OpenAI(api_key=api_key)

                    def generate_label(words_list):
                        prompt = f"Given the following words: {', '.join(words_list)}, generate a short, descriptive label for the topic."
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": prompt}]
                            )
                            return response.choices[0].message.content.strip()
                        except openai.RateLimitError:
                            st.error("Rate limit exceeded or insufficient quota. Please check your OpenAI billing/plan.")
                            return "LLM label unavailable"
                        except Exception as e:
                            st.error(f"OpenAI API error: {e}")
                            return "LLM label unavailable"

                    st.subheader("LLM-Generated Topic Labels:")
                    for idx, words in enumerate(topic_words):
                        label = generate_label(words)
                        st.write(f"**Topic {idx+1}:** {label}")
                        
                        log_topic_details(
                            dataset_name=uploaded_file.name,
                            method=method,
                            topic_num=idx + 1,
                            top_words=words,
                            llm_label=label,
                            processing_time=processing_time
                        )


else:
    st.info("Please upload a dataset to proceed.")


st.sidebar.markdown("### Session Log Viewer")
if st.sidebar.checkbox("Show Topic Modeling Logs"):
    log_file = "topic_modeling_logs.csv"
    if os.path.exists(log_file):
        log_data = pd.read_csv(log_file, on_bad_lines='skip')
        
        dataset_options = ["All"] + sorted(log_data["Dataset"].unique().tolist())
        method_options = ["All"] + sorted(log_data["Method"].unique().tolist())
        selected_dataset = st.sidebar.selectbox("Filter by Dataset", dataset_options)
        selected_method = st.sidebar.selectbox("Filter by Method", method_options)
        
        filtered_data = log_data.copy()
        if selected_dataset != "All":
            filtered_data = filtered_data[filtered_data["Dataset"] == selected_dataset]
        if selected_method != "All":
            filtered_data = filtered_data[filtered_data["Method"] == selected_method]
        
        if "Processing Time (s)" in filtered_data.columns:
            st.sidebar.write("⏱️ Processing Time (s) per run is included in the log below:")
        st.sidebar.dataframe(filtered_data)
    else:
        st.sidebar.info("No log data available yet.")


st.sidebar.markdown("### Topic Details Log Viewer")
if st.sidebar.checkbox("Show Topic Details Log"):
    detail_file = "topic_modeling_details.csv"
    if os.path.exists(detail_file):
        detail_data = pd.read_csv(detail_file, on_bad_lines='skip')
        dataset_options = ["All"] + sorted(detail_data["Dataset"].unique().tolist())
        method_options = ["All"] + sorted(detail_data["Method"].unique().tolist())
        selected_dataset = st.sidebar.selectbox("Filter Topic Log by Dataset", dataset_options)
        selected_method = st.sidebar.selectbox("Filter Topic Log by Method", method_options)
        if selected_dataset != "All":
            detail_data = detail_data[detail_data["Dataset"] == selected_dataset]
        if selected_method != "All":
            detail_data = detail_data[detail_data["Method"] == selected_method]
        st.sidebar.dataframe(detail_data)
    else:
        st.sidebar.info("No topic details log data available yet.")


st.sidebar.markdown("### Manage Logs")
if st.sidebar.button("Clear Logs"):
    try:
        if os.path.exists("topic_modeling_logs.csv"):
            os.remove("topic_modeling_logs.csv")
        if os.path.exists("topic_modeling_details.csv"):
            os.remove("topic_modeling_details.csv")
        st.sidebar.success("Log files cleared.")
    except Exception as e:
        st.sidebar.error(f"Error clearing logs: {e}")
