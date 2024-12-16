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

st.title("MIG Structured Sentiment")


###############################


import streamlit as st

# App Title
st.title("AI Sentiment Analysis Prompt Builder")

# Layout: Two columns
col1, col2 = st.columns(2)

# Left Column: User Inputs
with col1:
    st.header("Input Parameters")

    # Named Entity
    named_entity = st.text_input("Named Entity", placeholder="Enter the primary entity to analyze (e.g., Apple Inc.)")

    # Entity Variations
    entity_variations = st.text_area(
        "Entity Variations",
        placeholder="Enter other common variations of the entity name (e.g., Apple, AAPL), separated by commas.",
        help="List variations of the entity to ensure comprehensive analysis."
    )

    # Include Explanation
    include_explanation = st.checkbox(
        "Include Explanation",
        value=True,
        help="Check to include a 1-2 sentence explanation supporting the sentiment classification."
    )

    # Include Confidence Score
    include_confidence = st.checkbox(
        "Include Confidence Score",
        value=False,
        help="Check to include the confidence score of the sentiment analysis."
    )

    # Additional Guidance for Criteria
    additional_guidance = st.text_area(
        "Additional Sentiment Criteria",
        placeholder="Add specific instructions or criteria to guide the sentiment classification.",
        help="Optional: Provide any additional context or specific points to consider for sentiment evaluation."
    )

# Right Column: Prompt Preview
with col2:
    st.header("Prompt Preview")

    # Generate Prompt
    prompt = f"""
    You are an AI trained to perform expert-level sentiment analysis, with a keen awareness of how media stories can impact a brand's reputation. 
    You will analyze the sentiment of the following news story specifically for how it portrays {named_entity if named_entity else "[Named Entity Not Specified]"}. 
    Note that {named_entity if named_entity else "[Named Entity Not Specified]"} might be referred to by other common variations such as {entity_variations if entity_variations else "[No Variations Provided]"}. 
    Focus on how the organization is portrayed using the following criteria to guide your analysis:

    POSITIVE: Praises or highlights the {named_entity}'s achievements, contributions, or strengths.
    NEUTRAL: Provides balanced or factual coverage of the {named_entity} without clear positive or negative framing. Mentions the {named_entity} in a way that is neither supportive nor critical.
    NEGATIVE: Criticizes, highlights failures, or blames the {named_entity} for challenges or issues.

    {f"Additional Guidance: {additional_guidance}" if additional_guidance else ""}

    Note: Focus your analysis strictly on the sentiment toward {named_entity} rather than the broader topic or context of the story.

    {"Provide a single-word sentiment classification (POSITIVE, NEUTRAL, or NEGATIVE)" +
     (", followed by a colon and a one to two sentence explanation supporting your assessment." if include_explanation else ".")}
    {"Include a confidence score for your classification." if include_confidence else ""}
    If {named_entity} is not mentioned in the story, please reply with the phrase 'NOT RELEVANT'.

    Here is the story:
    """

    # Display Non-Editable Prompt
    st.text_area("Generated Prompt", prompt, height=400, disabled=True)



# Footer
st.markdown("---")
st.caption("Customize the inputs to dynamically generate a structured AI prompt for entity sentiment analysis.")

###############################


st.header("Custom Prompt:")

# Input custom prompt
custom_prompt = st.text_area("Enter your custom sentiment prompt here:", "")

# Row limit input
row_limit = st.number_input("Limit rows for testing (0 for all rows):", min_value=0, value=0, step=1)

df = st.session_state.unique_stories

# Filter rows without existing AI Sentiment values
if 'AI Sentiment' in df.columns:
    df = df[df['AI Sentiment'].isna() | (df['AI Sentiment'].str.len() == 0)]

# Apply row limit for the batch
if row_limit > 0:
    df = df.iloc[:row_limit]  # Select only the next batch of rows

st.write(f"Total Stories to Analyze: {len(df)}")

if st.button("Analyze Stories"):
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
                model="gpt-4o",
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

    # # Add the analysis results to the DataFrame
    # df['AI Sentiment'] = responses
    #
    # # Update the 'AI Sentiment' column in st.session_state.unique_stories
    # for _, row in df.iterrows():
    #     st.session_state.unique_stories.loc[
    #         st.session_state.unique_stories['Group ID'] == row['Group ID'], 'AI Sentiment'
    #     ] = row['AI Sentiment']

    # Add the analysis results to the DataFrame
    df['AI Sentiment'] = responses

    # # Split 'AI Sentiment' into 'AI Sentiment' and 'AI Sentiment Rationale' if a colon is present
    # df[['AI Sentiment', 'AI Sentiment Rationale']] = df['AI Sentiment'].str.split(':', 1, expand=True)

    # Ensure 'AI Sentiment' column contains only strings
    df['AI Sentiment'] = df['AI Sentiment'].astype(str)

    # df['AI Sentiment'] = df['AI Sentiment'].astype(str)

    # Split 'AI Sentiment' into 'AI Sentiment' and 'AI Sentiment Rationale' if a colon is present
    # df[['AI Sentiment', 'AI Sentiment Rationale']] = df['AI Sentiment'].str.split(':', 1, expand=True)
    df[['AI Sentiment', 'AI Sentiment Rationale']] = df['AI Sentiment'].str.split(':', n=1, expand=True)

    # Update the 'AI Sentiment' and 'AI Sentiment Rationale' columns in st.session_state.unique_stories
    for _, row in df.iterrows():
        st.session_state.unique_stories.loc[
            st.session_state.unique_stories['Group ID'] == row['Group ID'], 'AI Sentiment'
        ] = row['AI Sentiment']
        st.session_state.unique_stories.loc[
            st.session_state.unique_stories['Group ID'] == row['Group ID'], 'AI Sentiment Rationale'
        ] = row['AI Sentiment Rationale']


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
            st.session_state.df_traditional['Group ID'] == row['Group ID'], 'AI Sentiment'
        ] = row['AI Sentiment']
    # st.success("AI Sentiment cascaded to df_traditional successfully.")

# Add buttons for additional functionality
if st.button("Clear AI Sentiment"):
    st.session_state.unique_stories['AI Sentiment'] = None
    st.success("AI Sentiment column cleared successfully.")

# if st.button("Cascade AI Sentiment to df_traditional"):
#     for _, row in st.session_state.unique_stories.iterrows():
#         st.session_state.df_traditional.loc[
#             st.session_state.df_traditional['Group ID'] == row['Group ID'], 'AI Sentiment'
#         ] = row['AI Sentiment']
#     st.success("AI Sentiment cascaded to df_traditional successfully.")

