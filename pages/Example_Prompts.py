import streamlit as st


# Set Streamlit configuration
st.set_page_config(page_title="MIG Freeform Analysis Tool",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
                   layout="wide")


st.header("Example Prompts")

st.subheader("Custom Prompt Inputs", help="Customize the prompts with a specific brand name (and if needed, a list of relevant topics)")
col1, col2 = st.columns(2)
with col1:
    named_entity = st.text_input("Brand name for prompts", "")
    if len(named_entity) == 0:
        named_entity = "[BRAND]"

with col2:
    topic_list = st.text_input("Comma separated topic List", "", help="Only needed for Topic finder prompt")


st.subheader("Prompt Examples")
st.info("NOTE: These prompts are not perfect. They may not even be good. They are just examples to get you started.")


with st.expander("Product finder"):
    f"""
    Please analyze the following story to see if any {named_entity} products appear in it. 
    If yes, respond with only the list of names. If no, respond with just the word NO. Here is the story: 
    """


with st.expander("Spokesperson finder"):
    f"""
    Please analyze the following story to see if any {named_entity} spokespeople appear in it. 
    If yes, respond with only the list of names. If no, respond with just the word NO. Here is the story: 
    """


with st.expander("Topic finder"):
    f"""
    Please analyze the following story to see if {named_entity} is explicitly associated with any of the following topics in it:
    [{topic_list}].
    If yes, respond with only the list of topic names. If no, respond with just the word NO. Here is the story: 
    """


with st.expander("Sentiment & rationale"):
    f"""
    Analyze the sentiment of the following news story toward the {named_entity}. Focus on how the organization is portrayed using the following criteria to guide your analysis:\n
    POSITIVE: Praises or highlights the {named_entity}'s achievements, contributions, or strengths. 
    NEUTRAL: Provides balanced or factual coverage of the {named_entity} without clear positive or negative framing. Mentions the {named_entity} in a way that is neither supportive nor critical.
    NEGATIVE: Criticizes, highlights failures, or blames the {named_entity} for challenges or issues.
    Note: Focus your analysis strictly on the sentiment toward {named_entity} rather than the broader topic or context of the story. \n
    Provide a single-word sentiment classification (POSITIVE, NEUTRAL, or NEGATIVE) followed by a colon, then a one to two sentence explanation supporting your assessment. 
    If {named_entity} is not mentioned in the story, please reply with the phrase "NOT RELEVANT". Here is the story:
    """


with st.expander("Sentiment label only"):
    f"""
    Analyze the sentiment of the following news story toward the {named_entity}. Focus on how the organization is portrayed using the following criteria to guide your analysis:\n
    POSITIVE: Praises or highlights the {named_entity}'s achievements, contributions, or strengths. 
    NEUTRAL: Provides balanced or factual coverage of the {named_entity} without clear positive or negative framing. Mentions the {named_entity} in a way that is neither supportive nor critical.
    NEGATIVE: Criticizes, highlights failures, or blames the {named_entity} for challenges or issues.
    Note: Focus your analysis strictly on the sentiment toward {named_entity} rather than the broader topic or context of the story. \n
    Provide only a single-word sentiment classification: POSITIVE, NEUTRAL, NEGATIVE, or NOT RELEVANT (if {named_entity} is not mentioned in the story.
    Here is the story:
    """


with st.expander("Junk checker"):
    f"""
    Analyze the following news story or broadcast transcript to determine the type of coverage for the {named_entity}. Your response should be a single label from the following categories:\n
     - Press Release – The content appears to be directly from a press release or promotional material.\n
     - Advertisement – The brand mention is part of an advertisement or sponsored content.\n
     - Legitimate News – The brand is mentioned within a genuine news story or editorial context.\n
     Reply with only the category label that best fits the coverage.
    """