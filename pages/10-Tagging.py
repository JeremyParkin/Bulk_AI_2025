import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import concurrent.futures
from concurrent.futures import as_completed
import time
from threading import Lock
import json

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["key"])

# Set Streamlit config
st.set_page_config(
    page_title="AI Content Tagger",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide"
)

st.title("AI Tagging Tool")
st.warning("EXPERIMENTAL - use with caution")
st.info(
        "Generate AI tagging analysis for news stories based on the headline and full text of the story."
    )

client_name = st.session_state.client_name

# Ensure session state variables exist
st.session_state.setdefault("tag_definitions", {})
st.session_state.setdefault("tagging_mode", "Single best tag")

# Upload & config checks
if not st.session_state.upload_step:
    st.error('Please upload a CSV/XLSX before trying this step.')
    st.stop()
elif not st.session_state.config_step:
    st.error('Please run the configuration step before trying this step.')
    st.stop()

# Tagging configuration inputs (no longer inside a form)
st.write("**Define Tags and Criteria**")
tags_text = st.text_area(
    "One per line, in the format: TagName: Criteria",
    value=(
        f"Sustainability: {client_name} is discussed in relation to environmental responsibility, green initiatives, or emissions reduction\n"
        f"Innovation: {client_name} is discussed in relation to new technology, unique approaches, or product breakthroughs\n"
        f"Other: {client_name} is not discussed in relation to any other tagging topics."
    )
)
tagging_mode = st.radio("Tagging Mode", ["Single best tag", "Multiple applicable tags"])
model = st.selectbox("Select Model", ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"])

# Block execution if no definitions
if not st.session_state['tag_definitions']:
    st.warning("Please define at least one tag before running tagging.")

# Target data
start_row = 0
row_limit = st.number_input("Batch size (0 for all remaining rows):", min_value=0, value=5, step=1)

if "Processed" not in st.session_state.unique_stories.columns:
    st.session_state.unique_stories["Processed"] = False

unprocessed_df = st.session_state.unique_stories[~st.session_state.unique_stories["Processed"]]
if row_limit > 0:
    unprocessed_df = unprocessed_df.iloc[start_row:start_row + row_limit]
unprocessed_df = unprocessed_df.reset_index()
st.write(f"Selected Stories for Analysis: {len(unprocessed_df)}")

# Apply Tags
if st.button("Apply Tags", type='primary'):
    tag_definitions = {}
    for line in tags_text.strip().splitlines():
        if ':' in line:
            tag, criteria = line.split(':', 1)
            tag_definitions[tag.strip()] = criteria.strip()

    st.session_state['tag_definitions'] = tag_definitions
    st.session_state['tagging_mode'] = tagging_mode

    # Define function schemas based on tagging_mode
    if tagging_mode == "Single best tag":
        functions = [
            {
                "name": "apply_single_tag",
                "description": "Apply the best-fitting tag to a news story.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string", "description": "The single best tag"},
                        "explanation": {"type": "string", "description": "Why this tag was chosen"}
                    },
                    "required": ["tag", "explanation"]
                }
            }
        ]
    else:
        functions = [
            {
                "name": "apply_multiple_tags",
                "description": "Apply all relevant tags to a news story.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "All tags that apply"
                        },
                        "explanations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "One explanation per tag"
                        }
                    },
                    "required": ["tags", "explanations"]
                }
            }
        ]

    lock = Lock()
    progress_bar = st.progress(0)
    token_counts = {"input_tokens": 0, "output_tokens": 0}

    def analyze_story(row, tag_definitions, tagging_mode):
        snippet_column = "Coverage Snippet" if "Coverage Snippet" in unprocessed_df.columns else "Snippet"
        tag_rules = json.dumps(tag_definitions, indent=2)

        instruction = "Only return ONE tag. Do not return multiple. Even if several might apply, choose the ONE most relevant tag based on the criteria below. Return it as a single string, not as a list." if tagging_mode == "Single best tag" \
            else "Apply all tags that are relevant to the article."

        prompt = f"""
You are a media analysis AI. Your task is to apply tags to this story based on the definitions provided.

Tag Definitions:
{tag_rules}

Instructions:
{instruction}

Content:
Headline: {row['Headline']}
Snippet: {row[snippet_column]}
"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at tagging news stories based on defined criteria."},
                    {"role": "user", "content": prompt}
                ],
                functions=functions,
                function_call={"name": functions[0]["name"]}
            )
            args = json.loads(response.choices[0].message.function_call.arguments)

            with lock:
                token_counts["input_tokens"] += response.usage.prompt_tokens
                token_counts["output_tokens"] += response.usage.completion_tokens

            return args

        except Exception as e:
            return {"error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(analyze_story, row, tag_definitions, tagging_mode): idx
            for idx, row in unprocessed_df.iterrows()
        }

        for future in as_completed(futures):
            idx = futures[future]
            original_index = unprocessed_df.loc[idx, "index"]
            try:
                result = future.result()
                with lock:
                    if "error" not in result:
                        st.session_state.unique_stories.loc[original_index, "Processed"] = True

                        if tagging_mode == "Single best tag":
                            tag = result["tag"]
                            rationale = result["explanation"]

                            if isinstance(tag, str) and "," in tag:
                                rationale = "**NOTE: Multiple tags returned in single-tag mode.** " + rationale

                            st.session_state.unique_stories.loc[original_index, "AI Tag"] = tag
                            st.session_state.unique_stories.loc[original_index, "AI Tag Rationale"] = rationale
                        else:
                            st.session_state.unique_stories.loc[original_index, "AI Tags"] = ", ".join(result["tags"])
                            st.session_state.unique_stories.loc[original_index, "AI Tag Rationales"] = " | ".join(result["explanations"])
                            for tag in tag_definitions.keys():
                                col = f"AI Tag: {tag}"
                                st.session_state.unique_stories.loc[original_index, col] = 1 if tag in result["tags"] else 0
                    else:
                        st.session_state.unique_stories.loc[original_index, "AI Tag"] = f"Error: {result['error']}"

                progress_bar.progress((idx + 1) / len(unprocessed_df))

            except Exception as e:
                st.error(f"Error during tagging: {e}")

    progress_bar.progress(1.0)
    st.markdown(f"**Stories Analyzed:** {len(unprocessed_df)}")
    st.dataframe(st.session_state.unique_stories)

    in_tokens = token_counts["input_tokens"]
    out_tokens = token_counts["output_tokens"]
    if model == "gpt-4.1":
        cost = (in_tokens / 1_000_000) * 2.00 + (out_tokens / 1_000_000) * 8
    elif model == "gpt-4.1-mini":
        cost = (in_tokens / 1_000_000) * 0.40 + (out_tokens / 1_000_000) * 1.60
    elif model == "gpt-4.1-nano":
        cost = (in_tokens / 1_000_000) * 0.20 + (out_tokens / 1_000_000) * 0.80

    st.write(f"**Total Cost:** USD${cost:.4f}")

# Reset
if st.button("Reset Processed Rows"):
    st.session_state.unique_stories["Processed"] = False
    tag_columns_to_drop = [col for col in st.session_state.unique_stories.columns if col.startswith("AI Tag")]
    st.session_state.unique_stories.drop(columns=tag_columns_to_drop, inplace=True)
    st.success("All AI tag columns removed and rows marked unprocessed.")

with st.expander("Unique Stories"):
    st.dataframe(st.session_state.unique_stories)




# import streamlit as st
# import pandas as pd
# import openai
# from openai import OpenAI
# import concurrent.futures
# from concurrent.futures import as_completed
# import time
# from threading import Lock
# import json
#
# # Initialize OpenAI client
# client = OpenAI(api_key=st.secrets["key"])
#
# # Set Streamlit config
# st.set_page_config(page_title="AI Content Tagger",
#                    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
#                    layout="wide")
#
# st.title("AI Tagging Tool")
#
# client_name = st.session_state.client_name
#
# # Ensure session state variables exist
# st.session_state.setdefault("tag_definitions", {})
# st.session_state.setdefault("tagging_mode", "Single best tag")
#
# # Upload & config checks
# if not st.session_state.upload_step:
#     st.error('Please upload a CSV/XLSX before trying this step.')
#     st.stop()
# elif not st.session_state.config_step:
#     st.error('Please run the configuration step before trying this step.')
#     st.stop()
#
#
# st.subheader("Define Tags and Criteria")
# tags_text = st.text_area(
#     "One per line, in the format: TagName: Criteria",
#     value=(
#         f"Sustainability: {client_name} is discussed in relation to environmental responsibility, green initiatives, or emissions reduction\n"
#         f"Innovation: {client_name} is discussed in relation to new technology, unique approaches, or product breakthroughs\n"
#         f"Other: {client_name} is not discussed in relation to any other tagging topics."
#     )
# )
# tagging_mode = st.radio("Tagging Mode", ["Single best tag", "Multiple applicable tags"])
# model = st.selectbox("Select Model", ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"])
#
#
#
# # Step 1: User-defined tag input
# # with st.form("tag_definition_form"):
# #     st.subheader("Define Tags and Criteria")
# #     tags_text = st.text_area(
# #         "One per line, in the format: TagName: Criteria",
# #         value=(
# #             f"Sustainability: {client_name} is discussed in relation to environmental responsibility, green initiatives, or emissions reduction\n"
# #             f"Innovation: {client_name} is discussed in relation to new technology, unique approaches, or product breakthroughs\n"
# #             f"Other: {client_name} is not discussed in relation to any other tagging topics."
# #         ))
# #
# #     tagging_mode = st.radio("Tagging Mode", ["Single best tag", "Multiple applicable tags"])
# #     model = st.selectbox("Select Model", ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"])
# #     submitted = st.form_submit_button("Save Tagging Criteria")
# #
# # if submitted:
# #     tag_definitions = {}
# #     for line in tags_text.strip().splitlines():
# #         if ':' in line:
# #             tag, criteria = line.split(':', 1)
# #             tag_definitions[tag.strip()] = criteria.strip()
# #     st.session_state['tag_definitions'] = tag_definitions
# #     st.session_state['tagging_mode'] = tagging_mode
# #     st.success("Tag definitions saved.")
#
# # Block execution if no definitions
# if not st.session_state['tag_definitions']:
#     st.warning("Please define at least one tag before running tagging.")
#     st.stop()
#
# # Target data
# start_row = 0
# row_limit = st.number_input("Batch size (0 for all remaining rows):", min_value=0, value=5, step=1)
#
# if "Processed" not in st.session_state.unique_stories.columns:
#     st.session_state.unique_stories["Processed"] = False
#
# unprocessed_df = st.session_state.unique_stories[~st.session_state.unique_stories["Processed"]]
# if row_limit > 0:
#     unprocessed_df = unprocessed_df.iloc[start_row:start_row + row_limit]
# unprocessed_df = unprocessed_df.reset_index()
# st.write(f"Selected Stories for Analysis: {len(unprocessed_df)}")
#
# # Function schemas
# if st.session_state['tagging_mode'] == "Single best tag":
#     functions = [
#         {
#             "name": "apply_single_tag",
#             "description": "Apply the best-fitting tag to a news story.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "tag": {"type": "string", "description": "The single best tag"},
#                     "explanation": {"type": "string", "description": "Why this tag was chosen"}
#                 },
#                 "required": ["tag", "explanation"]
#             }
#         }
#     ]
# else:
#     functions = [
#         {
#             "name": "apply_multiple_tags",
#             "description": "Apply all relevant tags to a news story.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "tags": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                         "description": "All tags that apply"
#                     },
#                     "explanations": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                         "description": "One explanation per tag"
#                     }
#                 },
#                 "required": ["tags", "explanations"]
#             }
#         }
#     ]
#
# # Apply Tags
# if st.button("Apply Tags", type='primary'):
#     # Inside your Apply Tags block
#     tag_definitions = {}
#     for line in tags_text.strip().splitlines():
#         if ':' in line:
#             tag, criteria = line.split(':', 1)
#             tag_definitions[tag.strip()] = criteria.strip()
#
#     st.session_state['tag_definitions'] = tag_definitions
#     st.session_state['tagging_mode'] = tagging_mode
#
#
#     lock = Lock()
#     progress_bar = st.progress(0)
#     token_counts = {"input_tokens": 0, "output_tokens": 0}
#
#
#     def analyze_story(row, tag_definitions, tagging_mode):
#         snippet_column = "Coverage Snippet" if "Coverage Snippet" in unprocessed_df.columns else "Snippet"
#         tag_rules = json.dumps(tag_definitions, indent=2)
#
#         instruction = "Only return ONE tag. Do not return multiple. Even if several might apply, choose the ONE most relevant tag based on the criteria below. Return it as a single string, not as a list." if tagging_mode == "Single best tag" \
#             else "Apply all tags that are relevant to the article."
#
#         prompt = f"""
# You are a media analysis AI. Your task is to apply tags to this story based on the definitions provided.
#
# Tag Definitions:
# {tag_rules}
#
# Instructions:
# {instruction}
#
# Content:
# Headline: {row['Headline']}
# Snippet: {row[snippet_column]}
# """
#
#         try:
#             response = client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": "You are an expert at tagging news stories based on defined criteria."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 functions=functions,
#                 function_call={"name": "apply_single_tag"} if tagging_mode == "Single best tag"
#                 else {"name": "apply_multiple_tags"}
#             )
#             args = json.loads(response.choices[0].message.function_call.arguments)
#
#             with lock:
#                 token_counts["input_tokens"] += response.usage.prompt_tokens
#                 token_counts["output_tokens"] += response.usage.completion_tokens
#
#             return args
#
#         except Exception as e:
#             return {"error": str(e)}
#
#     with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#         futures = {
#             executor.submit(analyze_story, row, st.session_state['tag_definitions'],
#                             st.session_state['tagging_mode']): idx
#             for idx, row in unprocessed_df.iterrows()
#         }
#
#
#         for future in as_completed(futures):
#             idx = futures[future]
#             original_index = unprocessed_df.loc[idx, "index"]
#             try:
#                 result = future.result()
#                 with lock:
#                     if "error" not in result:
#                         st.session_state.unique_stories.loc[original_index, "Processed"] = True
#
#                         if st.session_state['tagging_mode'] == "Single best tag":
#                             tag = result["tag"]
#                             rationale = result["explanation"]
#
#                             if isinstance(tag, str) and "," in tag:
#                                 rationale = "**NOTE: Multiple tags returned in single-tag mode.** " + rationale
#
#                             st.session_state.unique_stories.loc[original_index, "AI Tag"] = tag
#                             st.session_state.unique_stories.loc[original_index, "AI Tag Rationale"] = rationale
#                             # st.session_state.unique_stories.loc[original_index, "AI Tag"] = result["tag"]
#                             # st.session_state.unique_stories.loc[original_index, "AI Tag Rationale"] = result["explanation"]
#                         else:
#                             st.session_state.unique_stories.loc[original_index, "AI Tags"] = ", ".join(result["tags"])
#                             st.session_state.unique_stories.loc[original_index, "AI Tag Rationales"] = " | ".join(result["explanations"])
#                             # Binary columns
#                             for tag in st.session_state['tag_definitions'].keys():
#                                 col = f"AI Tag: {tag}"
#                                 st.session_state.unique_stories.loc[original_index, col] = 1 if tag in result["tags"] else 0
#                     else:
#                         st.session_state.unique_stories.loc[original_index, "AI Tag"] = f"Error: {result['error']}"
#
#                 progress_bar.progress((idx + 1) / len(unprocessed_df))
#
#             except Exception as e:
#                 st.error(f"Error during tagging: {e}")
#
#     progress_bar.progress(1.0)
#
#     # Show results
#     st.markdown(f"**Stories Analyzed:** {len(unprocessed_df)}")
#     st.dataframe(st.session_state.unique_stories)
#
#     # Cost summary
#     in_tokens = token_counts["input_tokens"]
#     out_tokens = token_counts["output_tokens"]
#     if model == "gpt-4.1":
#         cost = (in_tokens / 1_000_000) * 2.00 + (out_tokens / 1_000_000) * 8
#     elif model == "gpt-4.1-mini":
#         cost = (in_tokens / 1_000_000) * 0.40 + (out_tokens / 1_000_000) * 1.60
#     elif model == "gpt-4.1-nano":
#         cost = (in_tokens / 1_000_000) * 0.20 + (out_tokens / 1_000_000) * 0.80
#
#     st.write(f"**Total Cost:** USD${cost:.4f}")
#
# # Reset
# if st.button("Reset Processed Rows"):
#     st.session_state.unique_stories["Processed"] = False
#
#     # Drop all columns starting with "AI Tag"
#     tag_columns_to_drop = [
#         col for col in st.session_state.unique_stories.columns
#         if col.startswith("AI Tag")
#     ]
#     st.session_state.unique_stories.drop(columns=tag_columns_to_drop, inplace=True)
#
#     st.success("All AI tag columns removed and rows marked unprocessed.")
#
#
# with st.expander("Unique Stories"):
#     st.dataframe(st.session_state.unique_stories)