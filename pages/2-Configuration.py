import streamlit as st
import pandas as pd
import mig_functions as mig
# import openai
# from openai import OpenAI
#
# client = OpenAI(api_key=st.secrets["key"])
import math
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_tags import st_tags

# Set Streamlit configuration
st.set_page_config(page_title="MIG Bulk AI Analysis Tool",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
                   layout="wide")

st.session_state.current_page = 'Configuration'

# Sidebar configuration
mig.standard_sidebar()


def normalize_text(text):
    """Convert to lowercase, remove extra spaces, remove punctuation, etc."""
    # Convert to string in case the input is not a string
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces from the beginning and end
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation (optional)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Further normalization steps like stemming could be added here
    return text


def remove_extra_spaces(text):
    """Remove extra spaces from the beginning and end of a string and replace multiple spaces with a single space."""
    # Convert to string in case the input is not a string
    text = str(text)
    # Remove extra spaces from the beginning and end
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text


def calculate_similarity(df):
    """Calculate the similarity between each article using TF-IDF and cosine similarity."""""
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Normalized Headline'] + " " + df['Normalized Snippet'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


def assign_group_ids(duplicates):
    """Assign a group ID to each article based on the similarity matrix."""
    group_id = 0
    group_ids = {}

    for i, similar_indices in duplicates.items():
        if i not in group_ids:
            group_ids[i] = group_id
            for index in similar_indices:
                group_ids[index] = group_id
            group_id += 1

    return group_ids


def identify_duplicates(similarity_matrix):
    """Identify duplicate articles based on the similarity matrix."""
    duplicates = {}
    for i in range(similarity_matrix.shape[0]):
        duplicates[i] = []
        for j in range(similarity_matrix.shape[1]):
            if i != j and similarity_matrix[i][j] > st.session_state.similarity_threshold:
                duplicates[i].append(j)
    return duplicates


def clean_snippet(snippet):
    """Remove the '>>>' or '>>' from braodcast snippets."""
    if snippet.startswith(">>>"):
        return snippet.replace(">>>", "", 1)
    if snippet.startswith(">>"):
        return snippet.replace(">>", "", 1)
    else:
        return snippet


st.title("Configuration")
if not st.session_state.upload_step:
    st.error('Please upload a CSV before trying this step.')


else:
    if not st.session_state.config_step:
        named_entity = st.session_state.client_name
        with st.form('User Inputs'):

            highlight_keyword = st_tags(
                label='Brand name variations to consider for prompts:',
                text='Press enter to add more',
                value=[st.session_state.client_name],
                maxtags=10,
                # key='1'
            )

            c1, c2 = st.columns(2, gap="large")
            with c1:
                random_sample = st.radio('Take a sample?', ['Yes, take a sample', 'No, use full data', ],
                                         help="Get a statistically significant random sample based on your uploaded data set.")

            # with c2:
            #     sentiment_opinion_selector = st.radio(
            #         "A.I. sentiment opinions?", ('Yes please', 'No thanks',),
            #         key='opinion_choice', help='Get GPT sentiment suggestions based on the article and client brand.')

            # with c3:
            #     sentiment_type = st.radio("Sentiment Type", ['3-way', '5-way'],
            #                               help='3-way is the standard approach.  5-way insteade uses *very positive*, *somewhat positive*, *neutral*, etc.')

            with c2:
                similarity_threshold = st.slider('Similarity level for grouping', min_value=0.85, value=0.9,
                                                 max_value=1.0,
                                                 help='Group articles that are the same or almost the same to be toned together.  0.85 is very similar, 1.0 is identical.')

            submitted = st.form_submit_button("Save Configuration", type="primary")

        if submitted:
            # st.session_state.df_traditional['Group ID'] = ''
            st.session_state.config_step = True

            # if sentiment_opinion_selector == 'Yes please':
            #     st.session_state.sentiment_opinion = True
            # else:
            #     st.session_state.sentiment_opinion = False

            st.session_state.random_sample = random_sample
            st.session_state.similarity_threshold = similarity_threshold
            # st.session_state.highlight_keyword = highlight_keyword

            # if sentiment_type == '3-way':
            #     st.session_state.sentiment_type = '3-way'
            #     # sentiment_instruction = f"""
            #     #     Please indicate the sentiment of the following news story as it pertains to {named_entity}.
            #     #     Start with one word: POSITIVE, NEUTRAL, or NEGATIVE - followed by a colon then a one sentence rationale
            #     #     as to why that sentiment was chosen. If {named_entity} is not mentioned in the story, please reply with
            #     #     the phrase "NOT RELEVANT: {named_entity} not mentioned in the story."
            #     #     """
            #     sentiment_instruction = f"""
            #         Instructions: Analyze the sentiment of the following news story specifically toward the named entity, {named_entity}. Focus on how the entity is portrayed using the following criteria to guide your analysis:
            #         Positive Sentiment: Praises or highlights the entity's achievements, contributions, or strengths. Uses favorable or supportive language toward the entity. Attributes beneficial or advantageous outcomes to the entity's actions.
            #         Neutral Sentiment: Provides balanced or factual coverage regarding the entity without clear positive or negative framing. Avoids strong language or bias in favor of or against the entity. Mentions the entity in a way that is neither supportive nor critical.
            #         Negative Sentiment: Criticizes, highlights failures, or associates the entity with challenges or issues. Uses unfavorable, disparaging, or hostile language toward the entity. Attributes negative outcomes or controversies to the entity's actions or decisions.
            #         IMPORTANT: Focus your analysis strictly on the sentiment toward {named_entity} rather than the broader topic or context of the story. Provide the uppercase sentiment classification (POSITIVE, NEUTRAL, or NEGATIVE) followed by a colon, then a one to two sentence explanation supporting your assessment.
            #         If {named_entity} is not mentioned in the story, please reply with the phrase "NOT RELEVANT: {named_entity} not mentioned in the story. Here is the story:
            #         """
            #
            # else:
            #     st.session_state.sentiment_type = '5-way'
            #     sentiment_instruction = f"""
            #         Instructions: Analyze the sentiment of the following news story specifically toward the named entity, {named_entity}. Focus on how the entity is portrayed using the following criteria to guide your analysis:
            #         Very Positive: Strongly praises the entity's achievements, contributions, or strengths. Uses highly favorable, supportive, or celebratory language toward the entity. Attributes significant beneficial outcomes or positive impacts to the entity's actions.
            #         Somewhat Positive: Highlights positive aspects of the entity or acknowledges strengths without strong praise. Uses moderately favorable language or frames the entity in a generally positive light. Attributes minor or moderate beneficial outcomes to the entity's actions.
            #         Neutral: Provides balanced or factual coverage regarding the entity without clear positive or negative framing. Avoids strong language or bias in favor of or against the entity. Mentions the entity in a way that is neither supportive nor critical.
            #         Somewhat Negative: Points out minor flaws, challenges, or criticisms related to the entity. Uses mildly unfavorable language or frames the entity in a somewhat negative light. Attributes minor or moderate negative outcomes to the entity's actions.
            #         Very Negative: Strongly criticizes the entity's actions, failures, or associations with significant issues. Uses highly unfavorable, disparaging, or hostile language toward the entity. Attributes substantial negative outcomes or impacts to the entity's actions.
            #         IMPORTANT: Focus your analysis strictly on the sentiment toward {named_entity} rather than the broader topic or context of the story. Provide the uppercase sentiment classification (VERY POSITIVE, SOMEWHAT POSITIVE, NEUTRAL, SOMEWHAT NEGATIVE, or VERY NEGATIVE) followed by a colon, then a one to two sentence explanation supporting your assessment.
            #         If {named_entity} is not mentioned in the story, please reply with the phrase "NOT RELEVANT: {named_entity} not mentioned in the story. Here is the story:
            #         """
            #
            #
            #     # sentiment_instruction = f"""
            #     #     Please indicate the sentiment of the following news story as it relates to {named_entity}.
            #     #     Start with the label: 'VERY POSITIVE', 'SOMEWHAT POSITIVE', 'NEUTRAL', 'SOMEWHAT NEGATIVE' or 'VERY NEGATIVE' -
            #     #     followed by a colon then a one sentence rationale as to why that sentiment was chosen. If {named_entity} is
            #     #     not mentioned in the story, please reply with the phrase "NOT RELEVANT: {named_entity} not mentioned in the story."
            #     #     """
            #
            # st.session_state.sentiment_instruction = sentiment_instruction
            #
            # if 'Assigned Sentiment' not in st.session_state.df_traditional.columns:
            #     st.session_state.df_traditional['Assigned Sentiment'] = pd.NA
            #
            # if 'Flagged for Review' not in st.session_state.df_traditional.columns:
            #     st.session_state.df_traditional['Flagged for Review'] = False

            # st.session_state.df_traditional = st.session_state.df_traditional.dropna(thresh=3)

            if random_sample == 'Yes, take a sample':
                def calculate_sample_size(N, confidence_level=0.95, margin_of_error=0.05, p=0.5):
                    # Z-score for 95% confidence level
                    Z = 1.96  # 95% confidence

                    numerator = N * (Z ** 2) * p * (1 - p)
                    denominator = (margin_of_error ** 2) * (N - 1) + (Z ** 2) * p * (1 - p)

                    return math.ceil(numerator / denominator)


                # Example usage
                population_size = len(st.session_state.full_dataset)
                sample_size = calculate_sample_size(population_size)

                st.session_state.sample_size = sample_size

                # Take a random sample of the DataFrame
                if sample_size < population_size:
                    df = st.session_state.df_traditional.sample(n=sample_size, random_state=1).reset_index(
                        drop=True)  # n=sample_size, random_state=1
                else:
                    # If the sample size is greater than or equal to the population, use the entire DataFrame
                    df = st.session_state.df_traditional.copy()

                st.write(f"Full data size: {len(st.session_state.df_traditional)}")
                st.write(f"Calculated sample size: {len(df)}")


            else:
                df = st.session_state.df_traditional

            if st.session_state.sentiment_opinion == True:
                # Ensure the "Sentiment Opinion" column exists in both dataframes
                if 'Sentiment Opinion' not in st.session_state.unique_stories.columns:
                    df['Sentiment Opinion'] = None

                if 'Sentiment Opinion' not in st.session_state.df_traditional.columns:
                    df['Sentiment Opinion'] = None

            df['Headline'] = df['Headline'].apply(remove_extra_spaces)
            df['Snippet'] = df['Snippet'].apply(remove_extra_spaces)
            df['Snippet'] = df['Snippet'].apply(clean_snippet)
            df['Normalized Headline'] = df['Headline'].apply(normalize_text)
            df['Normalized Snippet'] = df['Snippet'].apply(normalize_text)

            similarity_matrix = calculate_similarity(df)

            duplicate_groups = identify_duplicates(similarity_matrix)

            # Assign group IDs
            group_ids = assign_group_ids(duplicate_groups)
            df['Group ID'] = df.index.map(group_ids)

            # Drop the normalized columns
            df = df.drop(columns=['Normalized Headline', 'Normalized Snippet'])

            st.session_state.df_traditional = df.copy()

            # Calculate group counts
            group_counts = df.groupby('Group ID').size().reset_index(name='Group Count')

            # Group by 'Group ID' and keep the 'Group ID' column
            unique_stories = df.groupby('Group ID').agg(lambda x: x.iloc[0]).reset_index()

            # Merge group counts with unique_stories
            unique_stories_with_counts = unique_stories.merge(group_counts, on='Group ID')

            st.write('Unique Stories with Counts')
            st.write(unique_stories_with_counts)  # NO GROUP ID here

            # Sort unique stories by group count in descending order
            unique_stories_sorted = unique_stories_with_counts.sort_values(by='Group Count',
                                                                           ascending=False).reset_index(drop=True)

            # Update the session state
            st.session_state.unique_stories = unique_stories_sorted
            st.rerun()


    else:
        st.success('Configuration Completed!')

        st.write(f"Full data size: {len(st.session_state.full_dataset)}")
        st.write(f"Calculated sample size: {st.session_state.sample_size}")
        st.write(f"Unique stories in sample: {len(st.session_state.unique_stories)}")
        st.dataframe(st.session_state.unique_stories)


        def reset_config():
            st.session_state.config_step = False
            # Reset other relevant session state variables as needed
            st.session_state.sentiment_opinion = None
            st.session_state.random_sample = None
            st.session_state.similarity_threshold = None
            st.session_state.sentiment_instruction = None
            st.session_state.df_traditional = st.session_state.full_dataset.copy()
            st.session_state.counter = 0


        # Add reset button
        if st.button("Reset Configuration"):
            reset_config()
            st.rerun()  # Rerun the script to reflect the reset state
