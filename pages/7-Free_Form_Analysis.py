import streamlit as st
import pandas as pd
import openai
import io
import concurrent.futures
from openai import OpenAI
import time
from threading import Lock

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["key"])

# Set Streamlit configuration
st.set_page_config(
    page_title="MIG Freeform Analysis Tool",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
    layout="wide"
)

st.title("MIG Freeform Analysis")


st.subheader("Custom Prompt Inputs")


# Input custom prompt
custom_prompt = st.text_area("Enter your custom prompt here:", "")

col1, col2 = st.columns(2)
with col1:
    # Row limit input
    row_limit = st.number_input("Limit rows for testing (0 for all rows):", min_value=0, value=0, step=1)

df = st.session_state.unique_stories

# Filter rows without existing AI Analysis values
if 'Freeform Analysis' in df.columns:
    df = df[df['Freeform Analysis'].isna() | (df['Freeform Analysis'].str.len() == 0)]


# Apply row limit for the batch
if row_limit > 0:
    df = df.iloc[:row_limit]  # Select only the next batch of rows

st.write(f"Total Stories to Analyze: {len(df)}")


with col2:
    model = st.selectbox("Select Model", ["gpt-4o-mini", "gpt-4o"])




if st.button("Analyze Stories", type='primary'):
    openai.api_key = st.secrets["key"]
    responses = [None] * len(df)  # Initialize a list to store responses
    progress_bar = st.progress(0)  # Initialize the progress bar
    total_stories = len(df)

    token_counts = {"input_tokens": 0, "output_tokens": 0}  # Track token usage
    progress = {"completed": 0}  # Track progress
    lock = Lock()  # Thread-safe lock
    start_time = time.time()

    def update_progress():
        with lock:
            progress["completed"] += 1

    def analyze_story(row, index):
        snippet_column = "Coverage Snippet" if "Coverage Snippet" in df.columns else "Snippet"
        full_prompt = f"{custom_prompt}\n\n{row['Headline']}. {row[snippet_column]}"
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
                    {"role": "user", "content": full_prompt}
                ]
            )
            with lock:
                responses[index] = response.choices[0].message.content.strip()
                token_counts["input_tokens"] += response.usage.prompt_tokens
                token_counts["output_tokens"] += response.usage.completion_tokens
        except openai.OpenAIError as e:
            with lock:
                responses[index] = f"Error: {e}"
        update_progress()

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_index = {executor.submit(analyze_story, row, i): i for i, row in df.iterrows()}

        # Monitor progress
        while progress["completed"] < total_stories:
            completed = progress["completed"]
            progress_bar.progress(completed / total_stories)
            time.sleep(0.1)

        # Ensure progress bar reaches 100%
        progress_bar.progress(1.0)


    # Add the analysis results to the DataFrame
    df['Freeform Analysis'] = responses


    # Ensure 'Freeform Analysis' column contains only strings
    df['Freeform Analysis'] = df['Freeform Analysis'].astype(str)


    # Update the 'Freeform Analysis' column in st.session_state.unique_stories
    for _, row in df.iterrows():
        st.session_state.unique_stories.loc[
            st.session_state.unique_stories['Group ID'] == row['Group ID'], 'Freeform Analysis'
        ] = row['Freeform Analysis']



    end_time = time.time()
    elapsed_time = end_time - start_time

    # Display results
    st.markdown(
        f"**Stories Analyzed:** {total_stories} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Time Taken:** {elapsed_time:.2f} seconds",
        unsafe_allow_html=True
    )
    st.dataframe(df)

    # Display token usage
    st.markdown(
        f"**Total Input Tokens:** {token_counts['input_tokens']} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Total Output Tokens:** {token_counts['output_tokens']}",
        unsafe_allow_html=True
    )

    # Calculate and display costs
    total_input_tokens = token_counts['input_tokens']
    total_output_tokens = token_counts['output_tokens']
    input_cost = (total_input_tokens / 1_000_000) * 2.50  # Cost for input tokens
    output_cost = (total_output_tokens / 1_000_000) * 1.25  # Cost for output tokens
    total_cost = input_cost + output_cost

    st.markdown(
        f"**Cost for Input Tokens:** USD\${input_cost:.4f} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Cost for Output Tokens:** USD\${output_cost:.4f}",
        unsafe_allow_html=True
    )
    st.write(f"**Total Cost:** USD${total_cost:.4f}")

    for _, row in st.session_state.unique_stories.iterrows():
        st.session_state.df_traditional.loc[
            st.session_state.df_traditional['Group ID'] == row['Group ID'], 'Freeform Analysis'
        ] = row['Freeform Analysis']

# Add buttons for additional functionality
if st.button("Clear Freeform AI Analysis"):
    st.session_state.unique_stories['Freeform Analysis'] = None
    st.success("Freeform AI Analysis column cleared successfully.")

with st.expander("Unique Stories"):
    st.dataframe(st.session_state.unique_stories, hide_index=True)


st.divider()
st.subheader("Custom Prompt Examples")
st.divider()
# named_entity = st.text_input("Named Entity", "")
# if len(named_entity) == 0:
#     named_entity = "[BRAND]"
# topic_list = st.text_input("Comma seperated topic List", "")
# st.write("Update the prompt examples with the appropriate brand names and details for your use case.")
# st.info(
#     "NOTE: These prompts are not perfect. They may not even be good. They are just examples to get you started.")


named_entity = st.text_input("Named Entity", "")
if len(named_entity) == 0:
    named_entity = "[BRAND]"
topic_list = st.text_input("Comma seperated topic List", "")


with st.expander("Product finder"):
    f"""
    Please analyze the following story to see if any {named_entity} products appear in it. 
    If yes, respond with only the list of names. If no, respond with just the word 'No': 
    """

with st.expander("Spokesperson finder"):
    f"""
    Please analyze the following story to see if any {named_entity} spokespeople appear in it. 
    If yes, respond with only the list of names. If no, respond with just the word 'No': 
    """

with st.expander("Topic finder"):
    f"""
    Please analyze the following story to see if {named_entity} is explicitly associated with any of the following topics in it:
    [{topic_list}].
    If yes, respond with only the list of topic names. If no, respond with just the word 'No': 
    """

with st.expander("Sentiment"):
    f"""
    Analyze the sentiment of the following news story toward the {named_entity}. Focus on how the organization is portrayed using the following criteria to guide your analysis:\n
    POSITIVE: Praises or highlights the {named_entity}'s achievements, contributions, or strengths. 
    NEUTRAL: Provides balanced or factual coverage of the {named_entity} without clear positive or negative framing. Mentions the {named_entity} in a way that is neither supportive nor critical.
    NEGATIVE: Criticizes, highlights failures, or blames the {named_entity} for challenges or issues.
    Note: Focus your analysis strictly on the sentiment toward {named_entity} rather than the broader topic or context of the story. \n
    Provide a single-word sentiment classification (POSITIVE, NEUTRAL, or NEGATIVE) followed by a colon, then a one to two sentence explanation supporting your assessment. 
    If {named_entity} is not mentioned in the story, please reply with the phrase "NOT RELEVANT". Here is the story:
    """


with st.expander("Junk checker"):
    f"""
    Analyze the following news story or broadcast transcript to determine the type of coverage for the {named_entity}. Your response should be a single label from the following categories:\n
 - Press Release – The content appears to be directly from a press release or promotional material.\n
 - Advertisement – The brand mention is part of an advertisement or sponsored content.\n
 - Legitimate News – The brand is mentioned within a genuine news story or editorial context.\n
 Reply with only the category label that best fits the coverage.
    """

