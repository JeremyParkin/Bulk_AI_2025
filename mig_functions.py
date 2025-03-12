def standard_sidebar():
    import streamlit as st
    st.sidebar.image('https://app.agilitypr.com/app/assets/images/agility-logo-vertical.png', width=180)
    st.sidebar.subheader('MIG Bulk AI Analysis App')
    st.sidebar.caption("Version: Mar 2025")

    # CSS to adjust sidebar
    adjust_nav = """
                            <style>
                            .eczjsme9, .st-emotion-cache-1wqrzgl {
                                overflow: visible !important;
                                max-width: 250px !important;
                                }
                            .st-emotion-cache-a8w3f8 {
                                overflow: visible !important;
                                }
                            .st-emotion-cache-1cypcdb {
                                max-width: 250px !important;
                                }
                            </style>
                            """
    # Inject CSS with Markdown
    st.markdown(adjust_nav, unsafe_allow_html=True)

    # Add link to submit bug reports and feature requests
    st.sidebar.markdown(
        "[App Feedback](https://forms.office.com/Pages/ResponsePage.aspx?id=GvcJkLbBVUumZQrrWC6V07d2jCu79C5FsfEZJPZEfZxUNVlIVDRNNVBQVEgxQVFXNEM5VldUMkpXNS4u)")


# def top_x_by_mentions(df, column_name):
#     """Returns top 10 items by mention count"""
#     if not df[column_name].notna().any():
#         # If all values in the column are null, return an empty dataframe
#         return
#     top10 = df[[column_name, 'Mentions']].groupby(
#         by=[column_name]).sum().sort_values(
#         ['Mentions'], ascending=False)
#     top10 = top10.rename(columns={"Mentions": "Hits"})
#
#     return top10.head(10)
#
#
# def fix_author(df, headline_text, new_author):
#     """Updates all authors for a given headline"""
#     df.loc[df["Headline"] == headline_text, "Author"] = new_author
#
#
# def headline_authors(df, headline_text):
#     """Returns the various authors for a given headline"""
#     headline_authors = (df[df.Headline == headline_text].Author.value_counts().reset_index())
#     return headline_authors


def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f} B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f} M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f} K"
    else:
        return str(num)