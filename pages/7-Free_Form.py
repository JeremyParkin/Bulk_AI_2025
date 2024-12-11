import streamlit as st
import pandas as pd
import openai
import io
import concurrent.futures
from openai import OpenAI
import time
from threading import Lock

client = OpenAI(api_key=st.secrets["key"])

# Set Streamlit configuration
st.set_page_config(page_title="MIG Freeform Analysis Tool",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
                   layout="wide")


# Create a login
def check_password():
    """Returns True if the user had the correct username and password."""

    def credentials_entered():
        """Checks whether a username and password entered by the user are correct."""
        if st.session_state["username"] == st.secrets["USERNAME"] and st.session_state["password"] == st.secrets[
            "PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
            del st.session_state["username"]  # Don't store the username.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for username and password.
    st.text_input("Username", on_change=credentials_entered, key="username")
    st.text_input("Password", type="password", on_change=credentials_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("\ud83d\ude15 Username or password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

st.title("MIG Freeform Analysis Tool")
st.subheader("Experimental")

with st.expander("Instructions"):
    """
    The app will feed each story into the GPT-4 model to analyze the story based on the custom prompt you provide. 
    \nIt will merge in the HEADLINE and SNIPPET or COVERAGE SNIPPET fields following your custom prompt in order to analyze each story. 
    \nThe model will then generate a response for each story and the responses will be displayed in a table below. 
    \nYou can also download the results as an Excel file.
    \nUse cases could include identifying specific entities in news stories, associating stories to a list of categories, analyzing sentiment, or any other text-based analysis.
    """

with st.sidebar:
    st.header("Custom Prompt:")
    custom_prompt = st.text_area("Enter your analysis prompt here:",
                                 "",
                                 height=300)

    # Row limit input
    row_limit = st.number_input("Limit rows for testing (0 for all rows):", min_value=0, value=0, step=1)

upload_file = st.file_uploader("Upload a CSV or XLSX file:", type=["csv", "xlsx"])

if upload_file:
    # Check the file type
    file_type = upload_file.name.split('.')[-1]  # Get the file extension

    if file_type == "csv":
        df = pd.read_csv(upload_file)
    elif file_type == "xlsx":
        # If the file is an XLSX file, get the sheet names
        xls = pd.ExcelFile(upload_file)
        sheet_names = xls.sheet_names  # Get a list of all sheet names

        # Ask the user to select a sheet
        sheet = st.selectbox('Select a worksheet', sheet_names)

        # Read the selected sheet
        df = pd.read_excel(upload_file, sheet_name=sheet)

    if row_limit > 0:  # If user specifies a row limit greater than 0
        df = df.head(row_limit)  # Limit the DataFrame to the specified number of rows
    st.write(f"Total Stories: {len(df)}")

    if st.button("Analyze Stories"):
        openai.api_key = st.secrets["key"]
        responses = [None] * len(df)  # Initialize a list to store responses, indexed by row number
        progress_bar = st.progress(0)  # Initialize the progress bar
        total_stories = len(df)

        token_counts = {"input_tokens": 0, "output_tokens": 0}  # Use a dictionary to store token counts
        progress = {"completed": 0}  # Progress tracking
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

            # Main thread monitors progress
            while progress["completed"] < total_stories:
                completed = progress["completed"]
                progress_bar.progress(completed / total_stories)
                time.sleep(0.1)  # Adjust as needed to reduce UI lag

        # Ensure progress bar is set to 100% at the end
        progress_bar.progress(1.0)

        df['Analysis'] = responses

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Display the number of stories and elapsed time
        st.write(f"**Stories Analyzed:** {total_stories}")
        st.write(f"**Time Taken:** {elapsed_time:.2f} seconds")

        st.dataframe(df)

        # Display token usage
        st.write(f"**Total Input Tokens:** {token_counts['input_tokens']}")
        st.write(f"**Total Output Tokens:** {token_counts['output_tokens']}")

        # Define total input and output tokens
        total_input_tokens = token_counts['input_tokens']
        total_output_tokens = token_counts['output_tokens']

        # Calculate the costs
        input_cost = (total_input_tokens / 1_000_000) * 2.50  # Cost for input tokens
        output_cost = (total_output_tokens / 1_000_000) * 1.25  # Cost for output tokens
        total_cost = input_cost + output_cost

        # Display the costs
        st.write(f"**Cost for Input Tokens:** USD${input_cost:.4f}")
        st.write(f"**Cost for Output Tokens:** USD${output_cost:.4f}")
        st.write(f"**Total Cost:** USD${total_cost:.4f}")

        # Create a download link for the DataFrame as an Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
        output.seek(0)
        st.download_button(
            label="Download analysis results as Excel file",
            data=output,
            file_name="analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )




# import streamlit as st
# import pandas as pd
# import openai
# import io
# import concurrent.futures
# from openai import OpenAI
# import time
#
#
# client = OpenAI(api_key=st.secrets["key"])
#
# # Set Streamlit configuration
# st.set_page_config(page_title="MIG Freeform Analysis Tool",
#                    page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
#                    layout="wide")
#
# # Create a login
# def check_password():
#     """Returns `True` if the user had the correct username and password."""
#
#     def credentials_entered():
#         """Checks whether a username and password entered by the user are correct."""
#         if st.session_state["username"] == st.secrets["USERNAME"] and st.session_state["password"] == st.secrets["PASSWORD"]:
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # Don't store the password.
#             del st.session_state["username"]  # Don't store the username.
#         else:
#             st.session_state["password_correct"] = False
#
#     # Return True if the password is validated.
#     if st.session_state.get("password_correct", False):
#         return True
#
#     # Show input for username and password.
#     st.text_input("Username", on_change=credentials_entered, key="username")
#     st.text_input("Password", type="password", on_change=credentials_entered, key="password")
#     if "password_correct" in st.session_state:
#         st.error("\ud83d\ude15 Username or password incorrect")
#     return False
#
# if not check_password():
#     st.stop()  # Do not continue if check_password is not True.
#
# st.title("MIG Freeform Analysis Tool")
# st.subheader("Experimental")
#
# with st.expander("Instructions"):
#     """
#     The app will feed each story into the GPT-4 model to analyze the story based on the custom prompt you provide.
#     \nIt will merge in the HEADLINE and SNIPPET or COVERAGE SNIPPET fields following your custom prompt in order to analyze each story.
#     \nThe model will then generate a response for each story and the responses will be displayed in a table below.
#     \nYou can also download the results as an Excel file.
#     \nUse cases could include identifying specific entities in news stories, associating stories to a list of categories, analyzing sentiment, or any other text-based analysis.
#     """
#
# with st.sidebar:
#     st.header("Custom Prompt:")
#     custom_prompt = st.text_area("Enter your analysis prompt here:",
#                                  "",
#                                  height=300)
#
#     # Row limit input
#     row_limit = st.number_input("Limit rows for testing (0 for all rows):", min_value=0, value=0, step=1)
#
# upload_file = st.file_uploader("Upload a CSV or XLSX file:", type=["csv", "xlsx"])
#
# if upload_file:
#     # Check the file type
#     file_type = upload_file.name.split('.')[-1]  # Get the file extension
#
#     if file_type == "csv":
#         df = pd.read_csv(upload_file)
#     elif file_type == "xlsx":
#         # If the file is an XLSX file, get the sheet names
#         xls = pd.ExcelFile(upload_file)
#         sheet_names = xls.sheet_names  # Get a list of all sheet names
#
#         # Ask the user to select a sheet
#         sheet = st.selectbox('Select a worksheet', sheet_names)
#
#         # Read the selected sheet
#         df = pd.read_excel(upload_file, sheet_name=sheet)
#
#     if row_limit > 0:  # If user specifies a row limit greater than 0
#         df = df.head(row_limit)  # Limit the DataFrame to the specified number of rows
#     st.write(f"Total Stories: {len(df)}")
#
#     if st.button("Analyze Stories"):
#         openai.api_key = st.secrets["key"]
#         responses = [None] * len(df)  # Initialize a list to store responses, indexed by row number
#         progress_bar = st.progress(0)  # Initialize the progress bar
#         total_stories = len(df)
#
#         token_counts = {"input_tokens": 0, "output_tokens": 0}  # Use a dictionary to store token counts
#
#         start_time = time.time()
#
#         def analyze_story(row, index):
#             snippet_column = "Coverage Snippet" if "Coverage Snippet" in df.columns else "Snippet"
#             full_prompt = f"{custom_prompt}\n\n{row['Headline']}. {row[snippet_column]}"
#             try:
#                 response = client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=[
#                         {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
#                         {"role": "user", "content": full_prompt}
#                     ]
#                 )
#                 responses[index] = response.choices[0].message.content.strip()
#                 token_counts["input_tokens"] += response.usage.prompt_tokens
#                 token_counts["output_tokens"] += response.usage.completion_tokens
#             except openai.OpenAIError as e:
#                 responses[index] = f"Error: {e}"
#
#
#
#         # Use ThreadPoolExecutor for parallel processing
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future_to_index = {executor.submit(analyze_story, row, i): i for i, row in df.iterrows()}
#             for future in concurrent.futures.as_completed(future_to_index):
#                 i = future_to_index[future]
#                 try:
#                     future.result()
#                 except Exception as e:
#                     responses[i] = f"Error: {e}"
#                 progress_bar.progress(1 - (responses.count(None) / total_stories))
#
#         df['Analysis'] = responses
#
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#
#         # Display the number of stories and elapsed time
#         st.write(f"**Stories Analyzed:** {total_stories}")
#         st.write(f"**Time Taken:** {elapsed_time:.2f} seconds")
#
#
#         st.dataframe(df)
#
#
#         # Display token usage
#         st.write(f"**Total Input Tokens:** {token_counts['input_tokens']}")
#         st.write(f"**Total Output Tokens:** {token_counts['output_tokens']}")
#
#         # Define total input and output tokens
#         total_input_tokens = token_counts['input_tokens']
#         total_output_tokens = token_counts['output_tokens']
#
#         # Calculate the costs
#         input_cost = (total_input_tokens / 1_000_000) * 2.50  # Cost for input tokens
#         output_cost = (total_output_tokens / 1_000_000) * 1.25  # Cost for output tokens
#         total_cost = input_cost + output_cost
#
#         # Display the costs
#         st.write(f"**Cost for Input Tokens:** USD${input_cost:.4f}")
#         st.write(f"**Cost for Output Tokens:** USD${output_cost:.4f}")
#         st.write(f"**Total Cost:** USD${total_cost:.4f}")
#
#
#         # Create a download link for the DataFrame as an Excel file
#         output = io.BytesIO()
#         with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#             df.to_excel(writer, sheet_name='Sheet1', index=False)
#         output.seek(0)
#         st.download_button(
#             label="Download analysis results as Excel file",
#             data=output,
#             file_name="analysis_results.xlsx",
#             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#         )

# import streamlit as st
# import pandas as pd
# import openai
# import io
# import concurrent.futures
# from openai import OpenAI
#
# client = OpenAI(api_key=st.secrets["key"])
#
# # Set Streamlit configuration
# st.set_page_config(page_title="MIG Freeform Analysis Tool",
#                    page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
#                    layout="wide")
#
# # Create a login
# def check_password():
#     """Returns `True` if the user had the correct username and password."""
#
#     def credentials_entered():
#         """Checks whether a username and password entered by the user are correct."""
#         if st.session_state["username"] == st.secrets["USERNAME"] and st.session_state["password"] == st.secrets["PASSWORD"]:
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # Don't store the password.
#             del st.session_state["username"]  # Don't store the username.
#         else:
#             st.session_state["password_correct"] = False
#
#     # Return True if the password is validated.
#     if st.session_state.get("password_correct", False):
#         return True
#
#     # Show input for username and password.
#     st.text_input("Username", on_change=credentials_entered, key="username")
#     st.text_input("Password", type="password", on_change=credentials_entered, key="password")
#     if "password_correct" in st.session_state:
#         st.error("\ud83d\ude15 Username or password incorrect")
#     return False
#
# if not check_password():
#     st.stop()  # Do not continue if check_password is not True.
#
# st.title("MIG Freeform Analysis Tool")
# st.subheader("Experimental")
#
# with st.expander("Instructions"):
#     """
#     The app will feed each story into the GPT-4 model to analyze the story based on the custom prompt you provide.
#     \nIt will merge in the HEADLINE and COVERAGE SNIPPET fields following your prompt in order to analyze each story.
#     \nThe model will then generate a response for each story and the responses will be displayed in a table below.
#     \nYou can also download the results as an Excel file.
#     \nUse cases could include identifying specific entities in news stories, associating stories to a list of categories, analyzing sentiment, or any other text-based analysis.
#     """
#
# with st.sidebar:
#     st.header("Custom Prompt:")
#     custom_prompt = st.text_area("Enter your analysis prompt here:",
#                                  "Please analyze the following story to see if any Yamaha products appear in it. If yes, respond with only the list of names. If no, respond with just the word 'No': ",
#                                  height=250)
#
#     # Row limit input
#     row_limit = st.number_input("Limit rows for testing (0 for all rows):", min_value=0, value=0, step=1)
#
# upload_file = st.file_uploader("Upload a CSV or XLSX file:", type=["csv", "xlsx"])
#
# if upload_file:
#     # Check the file type
#     file_type = upload_file.name.split('.')[-1]  # Get the file extension
#
#     if file_type == "csv":
#         df = pd.read_csv(upload_file)
#     elif file_type == "xlsx":
#         # If the file is an XLSX file, get the sheet names
#         xls = pd.ExcelFile(upload_file)
#         sheet_names = xls.sheet_names  # Get a list of all sheet names
#
#         # Ask the user to select a sheet
#         sheet = st.selectbox('Select a worksheet', sheet_names)
#
#         # Read the selected sheet
#         df = pd.read_excel(upload_file, sheet_name=sheet)
#
#     if row_limit > 0:  # If user specifies a row limit greater than 0
#         df = df.head(row_limit)  # Limit the DataFrame to the specified number of rows
#     st.write(f"Total Stories: {len(df)}")
#
#     if st.button("Analyze Stories"):
#         openai.api_key = st.secrets["key"]
#         responses = [None] * len(df)  # Initialize a list to store responses, indexed by row number
#         progress_bar = st.progress(0)  # Initialize the progress bar
#         total_stories = len(df)
#
#         def analyze_story(row, index):
#             snippet_column = "Coverage Snippet" if "Coverage Snippet" in df.columns else "Snippet"
#             full_prompt = f"{custom_prompt}\n\n{row['Headline']}. {row[snippet_column]}"
#             try:
#                 response = client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=[
#                         {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
#                         {"role": "user", "content": full_prompt}
#                     ]
#                 )
#                 responses[index] = response.choices[0].message.content.strip()
#             except openai.OpenAIError as e:
#                 responses[index] = f"Error: {e}"
#
#         # Use ThreadPoolExecutor for parallel processing
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future_to_index = {executor.submit(analyze_story, row, i): i for i, row in df.iterrows()}
#             for future in concurrent.futures.as_completed(future_to_index):
#                 i = future_to_index[future]
#                 try:
#                     future.result()
#                 except Exception as e:
#                     responses[i] = f"Error: {e}"
#                 progress_bar.progress(1 - (responses.count(None) / total_stories))
#                 # progress_bar.progress((responses.count(None)) / total_stories)
#
#         df['Analysis'] = responses
#         st.dataframe(df)
#
#         # Create a download link for the DataFrame as an Excel file
#         output = io.BytesIO()
#         with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#             df.to_excel(writer, sheet_name='Sheet1', index=False)
#         output.seek(0)
#         st.download_button(
#             label="Download analysis results as Excel file",
#             data=output,
#             file_name="analysis_results.xlsx",
#             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#         )






# import streamlit as st
# import pandas as pd
# import openai
# import io
# from openai import OpenAI
# client = OpenAI(api_key=st.secrets["key"])
#
# # Set Streamlit configuration
# st.set_page_config(page_title="MIG Freeform Analysis Tool",
#                    page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
#                    layout="wide")
#
#
#
#
#
# # Create a login
# def check_password():
#     """Returns `True` if the user had the correct username and password."""
#
#     def credentials_entered():
#         """Checks whether a username and password entered by the user are correct."""
#         if st.session_state["username"] == st.secrets["USERNAME"] and st.session_state["password"] == st.secrets["PASSWORD"]:
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # Don't store the password.
#             del st.session_state["username"]  # Don't store the username.
#         else:
#             st.session_state["password_correct"] = False
#
#     # Return True if the password is validated.
#     if st.session_state.get("password_correct", False):
#         return True
#
#     # Show input for username and password.
#     st.text_input("Username", on_change=credentials_entered, key="username")
#     st.text_input("Password", type="password", on_change=credentials_entered, key="password")
#     if "password_correct" in st.session_state:
#         st.error("😕 Username or password incorrect")
#     return False
#
#
# if not check_password():
#     st.stop()  # Do not continue if check_password is not True.
#
#
#
# st.title("MIG Freeform Analysis Tool")
# st.subheader("Experimental")
#
#
#
# with st.expander("Instructions"):
#     """
#     The app will feed each story into the GPT-4 model to analyze the story based on the custom prompt you provide.
#     \nIt will merge in the HEADLINE and COVERAGE SNIPPET fields following your prompt in order to analyse each story.
#     \nThe model will then generate a response for each story and the responses will be displayed in a table below.
#     \nYou can also download the results as an Excel file.
#     \nUse cases could include identifying specific entities in news stories, associating stories to a list of categories, analyzing sentiment, or any other text-based analysis.
#     """
#
# with st.sidebar:
#     st.header("Custom Prompt:")
#     custom_prompt = st.text_area("Enter your analysis prompt here:",
#                                  "Please analyze the following story to see if any Yamaha products appear in it. If yes, respond with only the list of names. If no, respond with just the word 'No': ",
#                                  height=250)
#
#     # Row limit input
#     row_limit = st.number_input("Limit rows for testing (0 for all rows):", min_value=0, value=0, step=1)
#
# upload_file = st.file_uploader("Upload a CSV or XLSX file:", type=["csv", "xlsx"])
#
# if upload_file:
#     # Check the file type
#     file_type = upload_file.name.split('.')[1]  # Get the file extension
#
#     if file_type == "csv":
#         df = pd.read_csv(upload_file)
#     elif file_type == "xlsx":
#         # If the file is an XLSX file, get the sheet names
#         xls = pd.ExcelFile(upload_file)
#         sheet_names = xls.sheet_names  # Get a list of all sheet names
#
#         # Ask the user to select a sheet
#         sheet = st.selectbox('Select a worksheet', sheet_names)
#
#         # Read the selected sheet
#         df = pd.read_excel(upload_file, sheet_name=sheet)
#
#     if row_limit > 0:  # If user specifies a row limit greater than 0
#         df = df.head(row_limit)  # Limit the DataFrame to the specified number of rows
#     st.write(f"Total Stories: {len(df)}")
#
#
#     if st.button("Analyze Stories"):
#         openai.api_key = st.secrets["key"]
#         responses = []
#         progress_bar = st.progress(0)  # Initialize the progress bar
#         total_stories = len(df)
#
#         for i, row in df.iterrows():
#             snippet_column = "Coverage Snippet" if "Coverage Snippet" in df.columns else "Snippet"
#             full_prompt = f"{custom_prompt}\n\n{row['Headline']}. {row[snippet_column]}"
#
#         # for i, row in df.iterrows():
#         #     full_prompt = f"{custom_prompt}\n\n{row['Headline']}. {row['Coverage Snippet']}"
#             try:
#                 response = client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=[
#                         {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
#                         {"role": "user", "content": full_prompt}
#                     ]
#                 )
#                 responses.append(response.choices[0].message.content.strip())
#
#
#
#
#             except openai.OpenAIError as e:
#                 responses.append(f"Error: {e}")
#
#             # Update progress bar
#             progress_bar.progress((i + 1) / total_stories)
#
#         df['Analysis'] = responses  # Make sure this line is correctly indented
#         st.dataframe(df)
#
#         # Create a download link for the DataFrame as an Excel file
#         output = io.BytesIO()
#         with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#             df.to_excel(writer, sheet_name='Sheet1', index=False)
#         output.seek(0)
#         st.download_button(
#             label="Download analysis results as Excel file",
#             data=output,
#             file_name="analysis_results.xlsx",
#             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#         )
