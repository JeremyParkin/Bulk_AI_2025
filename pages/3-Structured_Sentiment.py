import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import concurrent.futures
from concurrent.futures import as_completed
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


# Main title of the page
st.title("AI Sentiment Analysis")

# Check if the upload step is complete
if not st.session_state.upload_step:
    st.error('Please upload a CSV/XLSX before trying this step.')

# Check if the configuration step is complete
elif not st.session_state.config_step:
    st.error('Please run the configuration step before trying this step.')

else:
    st.info(
        "This page will generate three new columns: 'AI Sentiment', 'AI Sentiment Confidence', and 'AI Sentiment Rationale'.")

    st.subheader("Sentiment Prompt Inputs")

    # Add input for named entity
    named_entity = st.text_input("**Full brand name for analysis**, e.g. *the Canada Mortgage and Housing Corporation (CMHC)*:", "", help='Enter the full brand name for analysis, e.g. *the Canada Mortgage and Housing Corporation (CHMC)*. If appropriate, include "the" before it and/or a common acronym in parentheses after.')
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
    toning_rationale = st.text_area("OPTIONAL: Provide any additional rationale, context, or guidance for the sentiment analysis:", "")

    # Ensure the "Processed" column exists
    if "Processed" not in st.session_state.unique_stories.columns:
        st.session_state.unique_stories["Processed"] = False

    # Inputs for start row and batch size
    start_row = 0
    row_limit = st.number_input("Batch size (0 for all remaining rows):", min_value=0, value=5, step=1)

    # Filter unprocessed rows
    unprocessed_df = st.session_state.unique_stories[~st.session_state.unique_stories["Processed"]]

    # Apply row limits
    if row_limit > 0:
        unprocessed_df = unprocessed_df.iloc[start_row:start_row + row_limit]
    else:
        unprocessed_df = unprocessed_df.iloc[start_row:]

    # Reset the index of the filtered DataFrame for proper indexing
    unprocessed_df = unprocessed_df.reset_index()  # Reset index to avoid mismatches

    st.write(f"Selected Stories for Analysis: {len(unprocessed_df)}")


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
        elif len(unprocessed_df) == 0:
            st.warning("No rows to process for sentiment analysis.")
        else:
            # Initialize lock
            lock = Lock()

            # Initialize responses and progress tracking
            responses = [{} for _ in range(len(unprocessed_df))]
            progress = {"completed": 0}
            progress_bar = st.progress(0)
            token_counts = {"input_tokens": 0, "output_tokens": 0}
            start_time = time.time()

            def analyze_story(row):
                snippet_column = "Coverage Snippet" if "Coverage Snippet" in unprocessed_df.columns else "Snippet"
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

                full_prompt = f"{custom_prompt}\n\nEntity: {named_entity}\nHeadline: {row['Headline']}. {row[snippet_column]} \n\n{contextual_instructions}"

                if toning_rationale.strip():
                    full_prompt += f"\nRationale: {toning_rationale}"

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
                            {"role": "user", "content": full_prompt}
                        ],
                        functions=functions,
                        function_call={"name": "analyze_sentiment"}  # Explicitly call the function
                    )

                    import json
                    function_args = json.loads(response.choices[0].message.function_call.arguments)

                    # Update token counts
                    with lock:
                        token_counts["input_tokens"] += response.usage.prompt_tokens
                        token_counts["output_tokens"] += response.usage.completion_tokens

                    return {
                        "sentiment": function_args['sentiment'],
                        "confidence": function_args['confidence'],
                        "explanation": function_args['explanation']
                    }

                except Exception as e:
                    return {"error": str(e)}

            # Process stories in parallel and update results
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(analyze_story, row): idx for idx, row in unprocessed_df.iterrows()}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        original_index = unprocessed_df.loc[idx, "index"]  # Map back to the original index

                        with lock:  # Thread-safe updates
                            if "error" not in result:
                                st.session_state.unique_stories.loc[
                                    original_index, ['AI Sentiment', 'AI Sentiment Confidence', 'AI Sentiment Rationale']
                                ] = result['sentiment'], result['confidence'], result['explanation']
                                st.session_state.unique_stories.loc[
                                    original_index, 'Processed'
                                ] = True
                            else:
                                st.session_state.unique_stories.loc[
                                    original_index, 'AI Sentiment'
                                ] = f"Error: {result['error']}"

                        # Update progress
                        with lock:
                            progress["completed"] += 1
                            progress_bar.progress(progress["completed"] / len(unprocessed_df))

                    except Exception as exc:
                        st.error(f"An error occurred: {exc}")

            # Ensure progress bar reaches 100%
            progress_bar.progress(1.0)

            # Display results
            st.markdown(f"**Stories Analyzed:** {len(unprocessed_df)}")
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
            output_cost = (total_output_tokens / 1_000_000) * 10  # Cost for output tokens
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



    # Add button to reset processed state
    if st.button("Reset Processed Rows"):
        st.session_state.unique_stories["Processed"] = False
        st.session_state.unique_stories['AI Sentiment'] = None
        st.session_state.unique_stories['AI Sentiment Confidence'] = None
        st.session_state.unique_stories['AI Sentiment Rationale'] = None
        st.session_state.df_traditional['AI Sentiment'] = None
        st.session_state.df_traditional['AI Sentiment Confidence'] = None
        st.session_state.df_traditional['AI Sentiment Rationale'] = None
        st.success("Reset all rows to unprocessed state.")


    with st.expander("Unique Stories"):
        st.dataframe(st.session_state.unique_stories)
