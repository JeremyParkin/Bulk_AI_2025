import streamlit as st
import pandas as pd
import openai
from openai import OpenAI

import concurrent.futures
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

st.title("AI Sentiment Analysis")

if not st.session_state.upload_step:
    st.error('Please upload a CSV/XLSX before trying this step.')

elif not st.session_state.config_step:
    st.error('Please run the configuration step before trying this step.')

else:
    st.info(
        "This page will generate three new columns: 'AI Sentiment', 'AI Sentiment Confidence', and 'AI Sentiment Rationale'.")

    st.subheader("Sentiment Prompt Inputs")
    # Add input for named entity
    named_entity = st.text_input("**Full brand name for analysis**, e.g. *the Canada Mortgage and Housing Corporation (CHMC)*:", "", help='Enter the full brand name for analysis, e.g. *the Canada Mortgage and Housing Corporation (CHMC)*. If appropriate, include "the" before it and/or a common acronym in parentheses after.')
    if named_entity.strip() == "":
        st.warning("Please enter a named entity for analysis.")

    custom_prompt = f"""
        Analyze the sentiment of the following news story toward {named_entity}. 
        Focus on how the organization is portrayed using the following criteria to guide your analysis:
        POSITIVE: Praises or highlights {named_entity}'s achievements, contributions, or strengths. 
        NEUTRAL: Provides balanced or factual coverage of {named_entity} without clear positive or negative framing. 
        Mentions {named_entity} in a way that is neither supportive nor critical. 
        NEGATIVE: Criticizes, highlights failures, or blames {named_entity} for challenges or issues. 
        NOT RELEVANT: If the story does not mention {named_entity} at all.
        Note: Focus your analysis strictly on the sentiment toward {named_entity} rather than the broader topic or context of the story.
        """

    # Add optional rationale input
    toning_rationale = st.text_area("Provide any rationale or context for the sentiment analysis (optional):", "")

    # Row limit input
    row_limit = st.number_input("Limit rows for testing (0 for all rows):", min_value=0, value=0, step=1)

    # Assume the dataframe is loaded into session state
    df = st.session_state.unique_stories

    # Ensure necessary columns exist
    for col in ['AI Sentiment', 'AI Sentiment Confidence', 'AI Sentiment Rationale']:
        if col not in st.session_state.unique_stories.columns:
            st.session_state.unique_stories[col] = None

    # Filter rows without existing AI Sentiment values
    if 'AI Sentiment' in df.columns:
        df = df[df['AI Sentiment'].isna() | (df['AI Sentiment'].str.len() == 0)]

    # Apply row limit for the batch
    if row_limit > 0:
        df = df.iloc[:row_limit]  # Select only the next batch of rows

    st.write(f"Total Stories to Analyze: {len(df)}")



    # Define the function schema for OpenAI
    functions = [
        {
            "name": "analyze_sentiment",
            "description": "Analyze the sentiment of a news piece toward a specific named entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "named_entity": {
                        "type": "string",
                        "description": "The specific brand or named entity being analyzed."
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"],
                        "description": "The sentiment label of the news piece toward the named entity."
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "The confidence score of the sentiment categorization as a percentage."
                    },
                    "explanation": {
                        "type": "string",
                        "description": "A 1-2 sentence explanation of why the sentiment label was chosen."
                    }
                },
                "required": ["named_entity", "sentiment", "confidence", "explanation"]
            }
        }
    ]

    if st.button("Analyze Stories", type='primary'):
        if not named_entity:
            st.error("Please provide a named entity for analysis.")
        else:
            # Initialize responses and progress tracking
            responses = [{} for _ in range(len(df))]
            progress_bar = st.progress(0)
            token_counts = {"input_tokens": 0, "output_tokens": 0}
            progress = {"completed": 0}
            lock = Lock()
            total_stories = len(df)
            start_time = time.time()

            def update_progress():
                with lock:
                    progress["completed"] += 1


            def analyze_story(row, index):
                snippet_column = "Coverage Snippet" if "Coverage Snippet" in df.columns else "Snippet"
                contextual_instructions = (
                    f"When analyzing sentiment toward {named_entity}, focus on how the coverage specifically describes or portrays {named_entity}. "
                    "Do not attribute sentiment about external topics, events, or conditions positive or negative to {named_entity}. For example:\n"
                    "1. If the article discusses research by {named_entity} on a negative topic, this does not inherently indicate negative sentiment toward {named_entity}.\n"
                    "2. A lecture series by {named_entity} on a tragic topic does not imply negative sentiment toward {named_entity}. Focus only on how {named_entity} is described or positioned in the article.\n\n"
                    "Example 1: \n"
                    "Entity: CMHC\n"
                    "Headline: \"CMHC Research Predicts Housing Market Downturn in 2024\"\n"
                    "Analysis: Although the housing market downturn is a negative topic, CMHC is not negatively portrayed in this story. The sentiment is neutral toward CMHC.\n\n"
                    "Example 2:\n"
                    "Entity: DePaul University\n"
                    "Headline: \"DePaul Hosts Lecture Series on Tragic Historical Events\"\n"
                    "Analysis: The topic of the lecture series is tragic, but DePaul University is portrayed positively as an institution promoting education and awareness. The sentiment is positive toward DePaul University.\n"
                    "When determining the confidence score for sentiment, evaluate based on the clarity of sentiment expressed toward {named_entity}:\n"
                    "1. Assign **90%-100%** if the sentiment is explicitly clear and unambiguous.\n"
                    "2. Assign **70%-89%** if the sentiment is clear but requires minor interpretation.\n"
                    "3. Assign **50%-69%** if the sentiment can be inferred but is not explicitly stated, or if there are mixed signals.\n"
                    "4. Assign **Below 50%** if the sentiment is ambiguous, indirect, or unrelated to {named_entity}."
                )

                full_prompt = f"{custom_prompt}\n\nEntity: {named_entity}\nHeadline: {row['Headline']}. {row[snippet_column]}\n\n{contextual_instructions}"
                if toning_rationale.strip():
                    full_prompt += f"\nRationale: {toning_rationale}"

                try:
                    st.write(f"Sending to API: {full_prompt}")  # Debug the prompt
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
                            {"role": "user", "content": full_prompt}
                        ],
                        functions=functions,
                        function_call={"name": "analyze_sentiment"}  # Explicitly call the function
                    )
                    # Debug response type and structure
                    st.write(f"Response type: {type(response)}")
                    st.write(f"Response dir: {dir(response)}")

                    # Adjust access to match response structure
                    function_args = eval(response.choices[0].message.function_call.arguments)
                    st.write(f"Function arguments: {function_args}")  # Debug the parsed function arguments

                    with lock:
                        responses[index] = {
                            "sentiment": function_args['sentiment'],
                            "confidence": function_args['confidence'],
                            "explanation": function_args['explanation']
                        }
                        token_counts["input_tokens"] += response.usage.prompt_tokens
                        token_counts["output_tokens"] += response.usage.completion_tokens

                except Exception as e:
                    st.write(f"API error: {e}")  # Debug any errors
                    with lock:
                        responses[index] = {"error": str(e)}
                update_progress()



            # Process stories in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_index = {executor.submit(analyze_story, row, i): i for i, row in df.iterrows()}

                # Monitor progress
                while progress["completed"] < total_stories:
                    progress_bar.progress(progress["completed"] / total_stories)
                    time.sleep(0.1)

                progress_bar.progress(1.0)  # Ensure progress bar reaches 100%



            # Update the DataFrame with responses
            for i, row in df.iterrows():
                response = responses[i]
                if "error" not in response:
                    st.session_state.unique_stories.loc[
                        st.session_state.unique_stories['Group ID'] == row['Group ID'], 'AI Sentiment'
                    ] = response['sentiment']
                    st.session_state.unique_stories.loc[
                        st.session_state.unique_stories['Group ID'] == row['Group ID'], 'AI Sentiment Confidence'
                    ] = response['confidence']
                    st.session_state.unique_stories.loc[
                        st.session_state.unique_stories['Group ID'] == row['Group ID'], 'AI Sentiment Rationale'
                    ] = response['explanation']
                else:
                    st.session_state.unique_stories.loc[
                        st.session_state.unique_stories['Group ID'] == row['Group ID'], 'AI Sentiment'
                    ] = f"Error: {response['error']}"

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Display results
            st.markdown(
                f"**Stories Analyzed:** {total_stories} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Time Taken:** {elapsed_time:.2f} seconds",
                unsafe_allow_html=True
            )
            st.dataframe(st.session_state.unique_stories)

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
            st.session_state.df_traditional['Group ID'] == row['Group ID'], ['AI Sentiment', 'AI Sentiment Confidence', 'AI Sentiment Rationale']
        ] = row[['AI Sentiment', 'AI Sentiment Confidence', 'AI Sentiment Rationale']].values



    # Add buttons for additional functionality
    if st.button("Clear AI Sentiment"):
        st.session_state.unique_stories['AI Sentiment'] = None
        st.session_state.unique_stories['AI Sentiment Confidence'] = None
        st.session_state.unique_stories['AI Sentiment Rationale'] = None
        st.session_state.df_traditional['AI Sentiment'] = None
        st.session_state.df_traditional['AI Sentiment Confidence'] = None
        st.session_state.df_traditional['AI Sentiment Rationale'] = None
        st.success("AI Sentiment column cleared successfully.")


    with st.expander("Unique Stories"):
        st.dataframe(st.session_state.unique_stories)