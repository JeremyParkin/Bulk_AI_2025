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

st.title("MIG Freeform Sentiment")
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

    # Split 'AI Sentiment' into 'AI Sentiment' and 'AI Sentiment Rationale' if a colon is present
    df[['AI Sentiment', 'AI Sentiment Rationale']] = df['AI Sentiment'].str.split(':', 1, expand=True)

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

# Add buttons for additional functionality
if st.button("Clear AI Sentiment"):
    st.session_state.unique_stories['AI Sentiment'] = None
    st.success("AI Sentiment column cleared successfully.")

if st.button("Cascade AI Sentiment to df_traditional"):
    for _, row in st.session_state.unique_stories.iterrows():
        st.session_state.df_traditional.loc[
            st.session_state.df_traditional['Group ID'] == row['Group ID'], 'AI Sentiment'
        ] = row['AI Sentiment']
    st.success("AI Sentiment cascaded to df_traditional successfully.")




# import streamlit as st
# import pandas as pd
# import openai
# import io
# import concurrent.futures
# from openai import OpenAI
# import time
# from threading import Lock
#
# client = OpenAI(api_key=st.secrets["key"])
#
# # Set Streamlit configuration
# st.set_page_config(page_title="MIG Freeform Analysis Tool",
#                    page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
#                    layout="wide")
#
# st.title("MIG Freeform Analysis Tool")
#
# st.header("Custom Prompt:")
# custom_prompt = st.text_area("Enter your analysis prompt here:",  "",)
#
# # Row limit input
# row_limit = st.number_input("Limit rows for testing (0 for all rows):", min_value=0, value=0, step=1)
#
# df = st.session_state.unique_stories
# if row_limit > 0:  # If user specifies a row limit greater than 0
#     df = df.head(row_limit)  # Limit the DataFrame to the specified number of rows
# st.write(f"Total Stories: {len(df)}")
#
# with st.expander("Show Data"):
#     st.dataframe(df)
#
#
# if st.button("Analyze Stories"):
#     openai.api_key = st.secrets["key"]
#     responses = [None] * len(df)  # Initialize a list to store responses, indexed by row number
#     progress_bar = st.progress(0)  # Initialize the progress bar
#     total_stories = len(df)
#
#     token_counts = {"input_tokens": 0, "output_tokens": 0}  # Use a dictionary to store token counts
#     progress = {"completed": 0}  # Progress tracking
#     lock = Lock()  # Thread-safe lock
#
#     start_time = time.time()
#
#
#     def update_progress():
#         with lock:
#             progress["completed"] += 1
#
#
#     def analyze_story(row, index):
#         snippet_column = "Coverage Snippet" if "Coverage Snippet" in df.columns else "Snippet"
#         full_prompt = f"{custom_prompt}\n\n{row['Headline']}. {row[snippet_column]}"
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
#                     {"role": "user", "content": full_prompt}
#                 ]
#             )
#             with lock:
#                 responses[index] = response.choices[0].message.content.strip()
#                 token_counts["input_tokens"] += response.usage.prompt_tokens
#                 token_counts["output_tokens"] += response.usage.completion_tokens
#         except openai.OpenAIError as e:
#             with lock:
#                 responses[index] = f"Error: {e}"
#         update_progress()
#
#
#     # Use ThreadPoolExecutor for parallel processing
#     with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#         future_to_index = {executor.submit(analyze_story, row, i): i for i, row in df.iterrows()}
#
#         # Main thread monitors progress
#         while progress["completed"] < total_stories:
#             completed = progress["completed"]
#             progress_bar.progress(completed / total_stories)
#             time.sleep(0.1)  # Adjust as needed to reduce UI lag
#
#     # Ensure progress bar is set to 100% at the end
#     progress_bar.progress(1.0)
#     st.info(
#         "NOTE: AI will make mistakes. Be sure to double check critical outputs!")
#
#     df['AI Sentiment'] = responses
#
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#
#     # Display the number of stories and elapsed time
#     st.markdown(
#         f"**Stories Analyzed:** {total_stories} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Time Taken:** {elapsed_time:.2f} seconds",
#         unsafe_allow_html=True
#     )
#
#
#     st.dataframe(df)
#
#     # Display token usage
#     st.markdown(
#         f"**Total Input Tokens:** {token_counts['input_tokens']} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Total Output Tokens:** {token_counts['output_tokens']}",
#         unsafe_allow_html=True
#     )
#
#     # Define total input and output tokens
#     total_input_tokens = token_counts['input_tokens']
#     total_output_tokens = token_counts['output_tokens']
#
#     # Calculate the costs
#     input_cost = (total_input_tokens / 1_000_000) * 2.50  # Cost for input tokens
#     output_cost = (total_output_tokens / 1_000_000) * 1.25  # Cost for output tokens
#     total_cost = input_cost + output_cost
#
#     # Display the costs
#     st.markdown(
#         f"**Cost for Input Tokens:** USD\\${input_cost:.4f} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Cost for Output Tokens:** USD\\${output_cost:.4f}",
#         unsafe_allow_html=True
#     )
#
#     st.write(f"**Total Cost:** USD${total_cost:.4f}")
#
#     # Create a download link for the DataFrame as an Excel file
#     output = io.BytesIO()
#     with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#         df.to_excel(writer, sheet_name='Sheet1', index=False)
#     output.seek(0)
#     st.download_button(
#         label="Download analysis results as Excel file",
#         data=output,
#         file_name="analysis_results.xlsx",
#         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#     )