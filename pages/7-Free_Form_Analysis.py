import streamlit as st
import pandas as pd
import openai
import io
import concurrent.futures
from concurrent.futures import as_completed

from openai import OpenAI
import time
from threading import Lock

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["key"])

# Set Streamlit configuration
st.set_page_config(
    page_title="AI Freeform Analysis",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide"
)

st.title("AI Freeform Analysis")

if not st.session_state.upload_step:
    st.error('Please upload a CSV/XLSX before trying this step.')

elif not st.session_state.config_step:
    st.error('Please run the configuration step before trying this step.')

else:
    st.info(
        "This page allows you to write your own prompt for any purpose you can imagine. Test carefully. Example prompts below.")

    st.subheader("Custom Prompt Inputs")

    # Input custom prompt
    custom_prompt = st.text_area("Enter your custom prompt here:", "")

    # Ensure the "FF_Processed" column exists
    if "FF_Processed" not in st.session_state.unique_stories.columns:
        st.session_state.unique_stories["FF_Processed"] = False

    # Inputs for start row and batch size
    start_row = 0


    col1, col2 = st.columns(2)
    with col1:
        # Row limit input
        row_limit = st.number_input("Limit rows for testing (0 for all rows):", min_value=0, value=0, step=1)

    # Filter unprocessed rows
    unprocessed_df = st.session_state.unique_stories[~st.session_state.unique_stories["FF_Processed"]]

    # Apply row limits
    if row_limit > 0:
        unprocessed_df = unprocessed_df.iloc[start_row:start_row + row_limit]
    else:
        unprocessed_df = unprocessed_df.iloc[start_row:]


    # Reset the index of the filtered DataFrame for proper indexing
    unprocessed_df = unprocessed_df.reset_index()  # Reset index to avoid mismatches

    st.write(f"Selected Stories for Analysis: {len(unprocessed_df)}")


    with col2:
        model = st.selectbox("Select Model", ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"])

    if st.button("Analyze Stories", type='primary'):
        if not custom_prompt:
            st.error("Please enter a custom prompt before proceeding.")
            # st.stop()
        elif len(unprocessed_df) == 0:
            st.success("All stories have been analyzed.")
            # st.stop()
        else:
            # Initialize lock
            lock = Lock()

            # Initialize responses and progress tracking
            responses = [{} for _ in range(len(unprocessed_df))]
            progress = {"completed": 0}
            progress_bar = st.progress(0)
            token_counts = {"input_tokens": 0, "output_tokens": 0}
            start_time = time.time()

            # Define a simplified function schema for OpenAI
            functions = [
                {
                    "name": "analyze_freeform",
                    "description": "Perform freeform analysis based on a user-defined prompt.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis": {
                                "type": "string",
                                "description": "The result of the analysis based on the user's custom prompt."
                            }
                        },
                        "required": ["analysis"]
                    }
                }
            ]



            def analyze_story(row):
                snippet_column = "Coverage Snippet" if "Coverage Snippet" in unprocessed_df.columns else "Snippet"
                full_prompt = f"{custom_prompt}\n\n{row['Headline']}. {row[snippet_column]}"

                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a highly adaptable AI for analyzing media."},
                            {"role": "user", "content": full_prompt}
                        ],
                        functions=functions,
                        function_call={"name": "analyze_freeform"}  # Explicitly call the function
                    )


                    if response.choices[0].message.function_call:
                        import json
                        function_args = json.loads(response.choices[0].message.function_call.arguments)

                        with lock:
                            token_counts["input_tokens"] += response.usage.prompt_tokens
                            token_counts["output_tokens"] += response.usage.completion_tokens


                        return {"analysis": function_args["analysis"]}

                except Exception as e:
                    return {"error": str(e)}

                return {"error": "No response received or analysis failed."}


            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(analyze_story, row): idx for idx, row in unprocessed_df.iterrows()}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        original_index = unprocessed_df.loc[idx, 'index']  # Map back to the original index

                        with lock:  # Thread-safe updates
                            if "error" not in result:
                                st.session_state.unique_stories.loc[original_index, 'Freeform Analysis'] = result[
                                    "analysis"]
                                st.session_state.unique_stories.loc[original_index, 'FF_Processed'] = True
                            else:
                                st.session_state.unique_stories.loc[
                                    original_index, 'Freeform Analysis'] = f"Error: {result['error']}"


                        with lock:
                            progress["completed"] += 1
                            progress_bar.progress(progress["completed"] / len(unprocessed_df))

                    except Exception as exc:
                        st.error(f"An error occurred: {exc}")

            # Ensure progress bar reaches 100%
            progress_bar.progress(1.0)


            # Add the analysis results to the DataFrame
            unprocessed_df['Freeform Analysis'] = responses

            # Ensure 'Freeform Analysis' column contains only strings
            unprocessed_df['Freeform Analysis'] = unprocessed_df['Freeform Analysis'].astype(str)


            # Display results
            st.markdown(
                f"**Stories Analyzed:** {len(unprocessed_df)}") # &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Time Taken:** {elapsed_time:.2f} seconds",
                # unsafe_allow_html=True )

            # Determine the snippet column name dynamically
            snippet_column = "Coverage Snippet" if "Coverage Snippet" in st.session_state.unique_stories.columns else "Snippet"

            # Display the DataFrame with the selected snippet column
            (st.dataframe(
                st.session_state.unique_stories[['Headline', snippet_column, 'Freeform Analysis']]))
            # ,)
            #     hide_index=True
            # )

            # st.dataframe(st.session_state.unique_stories[['Headline', 'Freeform Analysis']], hide_index=True)

            # Display token usage
            st.markdown(
                f"**Total Input Tokens:** {token_counts['input_tokens']} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Total Output Tokens:** {token_counts['output_tokens']}",
                unsafe_allow_html=True
            )

            # Calculate and display costs
            total_input_tokens = token_counts['input_tokens']
            total_output_tokens = token_counts['output_tokens']

            # model = st.selectbox("Select Model", ["gpt-4o-mini", "gpt-4o"])
            if model == "gpt-4o":
                input_cost = (total_input_tokens / 1_000_000) * 2.50  # Cost for input tokens
                output_cost = (total_output_tokens / 1_000_000) * 10  # Cost for output tokens

            if model == "gpt-4.1":
                input_cost = (total_input_tokens / 1_000_000) * 2.0  # Cost for input tokens
                output_cost = (total_output_tokens / 1_000_000) * 8  # Cost for output tokens

            if model == "gpt-4.1-mini":
                input_cost = (total_input_tokens / 1_000_000) * 0.40  # Cost for input tokens
                output_cost = (total_output_tokens / 1_000_000) * 1.60  # Cost for output tokens

            else:
                input_cost = (total_input_tokens / 1_000_000) * 0.15  # Cost for input tokens
                output_cost = (total_output_tokens / 1_000_000) * 0.60  # Cost for output tokens
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
        st.session_state.unique_stories['FF_Processed'] = False
        st.session_state.unique_stories['Freeform Analysis'] = None
        st.success("Freeform AI Analysis column cleared successfully.")

    with st.expander("Unique Stories"):
        st.dataframe(st.session_state.unique_stories, hide_index=True)

    st.divider()
    st.header("Example Prompts")
    st.info(
        "NOTE: These prompts are not perfect. They may not even be good. They are just examples to get you started.")

    st.subheader("Custom Prompt Inputs",
                 help="Customize the prompts with a specific brand name (and if needed, a list of relevant topics)")
    col1, col2 = st.columns(2)
    with col1:
        named_entity = st.text_input("Brand name for prompts", f"{st.session_state.client_name}")
        if len(named_entity) == 0:
            named_entity = "[BRAND]"

    with col2:
        topic_list = st.text_input("Comma separated topic List", "", help="Only needed for Topic finder prompt")

    st.subheader("Prompt Examples")


    with st.expander("Product finder"):
        f"""
        Please analyze the following story to see if any {named_entity} products appear in it.
        If yes, respond with the list of product names separated by commas, and nothing else. If no, respond with just the word NO. Here is the story:
        """

    with st.expander("Spokesperson finder"):
        f"""
        Please analyze the following story to see if any {named_entity} representatives or spokespeople appear in it.
        If yes, respond only with the list of their names separated by commas. If no, respond with just the word NO. Here is the story:
        """


    with st.expander("Topic finder"):
        f"""
        Please analyze the following story to see if {named_entity} is explicitly associated with any of the following topics in it:
        [{topic_list}].
        If yes, respond with the list of topic names separated by commas, and nothing else. If no, respond with just the word NO. Here is the story:
        """

    # with st.expander("Sentiment & rationale"):
    #     f"""
    #     Analyze the sentiment of the following news story toward {named_entity}. Focus on how the organization is portrayed using the following criteria to guide your analysis:\n
    #     POSITIVE: Praises or highlights {named_entity}'s achievements, contributions, or strengths.
    #     NEUTRAL: Provides balanced or factual coverage of {named_entity} without clear positive or negative framing. Mentions {named_entity} in a way that is neither supportive nor critical.
    #     NEGATIVE: Criticizes, highlights failures, or blames {named_entity} for challenges or issues.
    #     Note: Focus your analysis strictly on the sentiment toward {named_entity} rather than the broader topic or context of the story. \n
    #     Provide a single-word sentiment classification (POSITIVE, NEUTRAL, or NEGATIVE) followed by a colon, then a one to two sentence explanation supporting your assessment.
    #     If {named_entity} is not mentioned in the story, please reply with the phrase "NOT RELEVANT". Here is the story:
    #     """
    #
    # with st.expander("Sentiment label only"):
    #     f"""
    #     Analyze the sentiment of the following news story toward {named_entity}. Focus on how the organization is portrayed using the following criteria to guide your analysis:\n
    #     POSITIVE: Praises or highlights {named_entity}'s achievements, contributions, or strengths.
    #     NEUTRAL: Provides balanced or factual coverage of {named_entity} without clear positive or negative framing. Mentions {named_entity} in a way that is neither supportive nor critical.
    #     NEGATIVE: Criticizes, highlights failures, or blames {named_entity} for challenges or issues.
    #     Note: Focus your analysis strictly on the sentiment toward {named_entity} rather than the broader topic or context of the story. \n
    #     Provide only a single-word sentiment classification: POSITIVE, NEUTRAL, NEGATIVE, or NOT RELEVANT (if {named_entity} is not mentioned in the story.
    #     Here is the story:
    #     """

    with st.expander("Junk checker"):
        f"""
        Analyze the following news story or broadcast transcript to determine the type of coverage for [BRAND]. Your response should be a single label from the following categories:
        •Press Release – The content appears to be directly from a press release or promotional material.
        •Advertisement – The brand mention is part of an advertisement or sponsored content.
        •Event Listing / Calendar Notice – A public announcement of an event where the brand is mentioned but not discussed further.
        •Stock Market Update – A brief stock price update with no meaningful discussion of the company.
        •Job Posting – The content is a recruitment ad mentioning the brand in an employment context.
        •Incidental Bio Mention – The brand is mentioned only in a person’s biography or background information, such as an author’s credentials or a public figure’s past employment history, without any discussion or relevance to the brand in the story itself.
        •Legitimate News – The brand is discussed within a genuine news story or editorial context.
        Reply with only the category label that best fits the coverage. Here is the story:
        """