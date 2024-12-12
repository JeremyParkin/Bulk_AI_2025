import mig_functions as mig
import pickle
import base64
import streamlit as st
import pandas as pd
import io
from datetime import datetime



# Set Streamlit configuration
st.set_page_config(page_title="MIG Sentiment App",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
                   layout="wide")

# Sidebar configuration
mig.standard_sidebar()

st.session_state.current_page = 'Save and Load'

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M")



def save_session_state():
    # Manually copy necessary items from session state
    session_data = {key: value for key, value in st.session_state.items()}

    # Convert DataFrames to CSV strings
    for df_name in ['df_traditional', 'unique_stories', 'full_dataset']:
        if df_name in session_data:
            buffer = io.StringIO()
            session_data[df_name].to_csv(buffer, index=False)
            session_data[df_name] = buffer.getvalue()

    # Serialize the session state
    serialized_data = pickle.dumps(session_data)

    # Encode the serialized data for downloading
    b64 = base64.b64encode(serialized_data).decode()

    # Generate a download link
    href = f'<a href="data:file/pkl;base64,{b64}" download="{st.session_state.client_name} - {dt_string}.pkl">Download Session State</a>'
    return href



def load_session_state(uploaded_file):
    if uploaded_file is not None:
        # Read the uploaded file
        session_data = uploaded_file.getvalue()

        # Deserialize the session state
        deserialized_data = pickle.loads(session_data)

        # Convert CSV strings back to DataFrames
        for df_name in ['df_traditional', 'unique_stories', 'full_dataset']:
            if df_name in deserialized_data:
                buffer = io.StringIO(deserialized_data[df_name])
                deserialized_data[df_name] = pd.read_csv(buffer)

        # Update the session state
        st.session_state.update(deserialized_data)

        st.success("Session state loaded successfully!")


st.title("Save & Load")

if not st.session_state.upload_step:
    st.error('Please upload a CSV before SAVING.')

elif not st.session_state.config_step:
    st.error('Please run the configuration step before SAVING.')

else:
    st.divider()

    st.header("Save")
    st.info("**SAVE** your current data-processing session to a downloadable .pkl file")

    # When this button is clicked, the save_session_state function will be executed
    if st.button("Generate Session File to Download"):
        # Generate the download link (or any other way you handle the saving)
        href = save_session_state()

        st.markdown(href, unsafe_allow_html=True)

    st.divider()

st.header("Load")
st.info("**LOAD** a previously saved data-processing session from a downloaded .pkl file")
uploaded_file = st.file_uploader("Upload Session State", type="pkl")
if uploaded_file is not None:
    load_session_state(uploaded_file)