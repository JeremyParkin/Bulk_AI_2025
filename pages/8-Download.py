import streamlit as st
import pandas as pd
import mig_functions as mig
import io
import altair as alt


# Set Streamlit configuration
st.set_page_config(page_title="MIG Sentiment Tool",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
                   layout="wide")


# Sidebar configuration
mig.standard_sidebar()

st.title("Download")


if st.session_state.get('current_page') != 'Download':
    st.session_state.download_data = None


st.session_state.current_page = 'Download'

if not st.session_state.upload_step:
    st.error('Please upload a CSV/XLSX before trying this step.')
elif not st.session_state.config_step:
    st.error('Please run the configuration step before trying this step.')
else:
    if 'AI Sentiment' in st.session_state.df_traditional.columns:

        if pd.notna(st.session_state.df_traditional['AI Sentiment']).any():
            # Count the frequency of each sentiment and calculate the percentage
            sentiment_counts = st.session_state.df_traditional['AI Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            total = sentiment_counts['Count'].sum()
            sentiment_counts['Percentage'] = (sentiment_counts['Count'] / total)

            if st.session_state.sentiment_type == '5-way':
                custom_order = ['VERY POSITIVE', 'SOMEWHAT POSITIVE', 'NEUTRAL', 'SOMEWHAT NEGATIVE', 'VERY NEGATIVE', 'NOT RELEVANT']
            else:
                custom_order = ['POSITIVE', 'NEUTRAL', 'NEGATIVE', 'NOT RELEVANT']

            # Apply the custom order to 'Assigned Sentiment' in df_traditional
            st.session_state.df_traditional['AI Sentiment'] = pd.Categorical(
                st.session_state.df_traditional['AI Sentiment'], categories=custom_order, ordered=True)


            # Ensure all sentiments are in the dataframe (to avoid key errors in sorting)
            for sentiment in custom_order:
                if sentiment not in sentiment_counts['Sentiment'].values:
                    # Create a new DataFrame for the missing sentiment and concatenate it
                    missing_sentiment_df = pd.DataFrame({'Sentiment': [sentiment], 'Count': [0], 'Percentage': [0.0]})
                    sentiment_counts = pd.concat([sentiment_counts, missing_sentiment_df], ignore_index=True)



            # Sort the dataframe based on the custom order
            sentiment_counts['Sentiment'] = pd.Categorical(sentiment_counts['Sentiment'], categories=custom_order, ordered=True)

            st.subheader('Sentiment Statistics')

            col1, col2 = st.columns([3,2], gap='large')
            with col1:
                color_mapping = {
                    "POSITIVE": "green",
                    "NEUTRAL": "yellow",
                    "NEGATIVE": "red",
                    "VERY POSITIVE": "darkgreen",
                    "SOMEWHAT POSITIVE": "limegreen",
                    "SOMEWHAT NEGATIVE": "coral",
                    "VERY NEGATIVE": "maroon",
                    "NOT RELEVANT": "dimgray"
                }

                # Create the color domain and range from the sorted DataFrame
                color_domain = custom_order
                color_range = [color_mapping.get(sentiment, "grey") for sentiment in color_domain]

                # Create the color scale
                color_scale = alt.Scale(domain=color_domain, range=color_range)


                # Create the base chart for the horizontal bar chart
                base = alt.Chart(sentiment_counts).encode(
                    y=alt.Y('Sentiment:N', sort=custom_order),
                )

                # Create the bar chart
                bar_chart = base.mark_bar().encode(
                    x='Count:Q',
                    color=alt.Color('Sentiment:N', scale=color_scale, legend=None),
                    tooltip=['Sentiment', alt.Tooltip('Percentage', format='.1f', title='Percent'), 'Count']
                )

                # Create the text labels for the bars
                text = base.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3,  # Nudges text to the right so it doesn't appear on top of the bar
                    color='ivory'  # Set text color to off-white
                ).encode(
                    x=alt.X('Count:Q', axis=alt.Axis(title='Mentions')),
                    text=alt.Text('Percentage:N', format='.1%')
                )

                # Combine the bar and text charts
                chart = (bar_chart + text).properties(
                    # title='Sentiment Bar Chart',
                    width=600,
                    height=250
                ).configure_view(
                    strokeWidth=0
                ).configure_axisLeft(
                    labelLimit=180  # Increase label limit to accommodate longer labels
                )

                st.altair_chart(chart, use_container_width=True)



            with col2:

                # Adding percentage column to the stats table
                sentiment_stats = sentiment_counts.copy()


                # Convert 'Sentiment' to a categorical type with the custom order
                sentiment_counts['Sentiment'] = pd.Categorical(sentiment_counts['Sentiment'], categories=custom_order,
                                                               ordered=True)

                # Sort 'sentiment_counts' by the categorical order
                sentiment_counts.sort_values(by='Sentiment', inplace=True)

                sentiment_counts['Percentage'] = (sentiment_counts['Percentage'] * 100).apply(lambda x: "{:.1f}%".format(x))


                # Display the table without the index
                st.dataframe(sentiment_counts, hide_index=True,)



    with st.expander('View Processed Data'):
        st.dataframe(st.session_state.df_traditional)
        # st.dataframe(st.session_state.df_traditional.loc[:,
        #              ~st.session_state.df_traditional.columns.isin(["FF_Processed", "Processed"])])


    traditional = st.session_state.df_traditional


    # Initialize a session state variable for the download link
    if 'download_data' not in st.session_state:
        st.session_state.download_data = None


    with st.form("my_form_download"):
        st.subheader("Generate your cleaned data workbook")
        submitted = st.form_submit_button("Go!", type="primary")

        if submitted:
            with st.spinner('Building workbook now...'):
                output = io.BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter', datetime_format='yyyy-mm-dd')

                workbook = writer.book

                # Add some cell formats.
                number_format = workbook.add_format({'num_format': '#,##0'})
                currency_format = workbook.add_format({'num_format': '$#,##0'})

                if len(traditional) > 0:
                    traditional = traditional.sort_values(by=['Impressions'], ascending=False)
                    traditional.to_excel(writer, sheet_name='CLEAN TRAD', startrow=1, header=False, index=False)
                    worksheet1 = writer.sheets['CLEAN TRAD']
                    worksheet1.set_tab_color('black')

                    (max_row, max_col) = traditional.shape
                    column_settings = [{'header': column} for column in traditional.columns]
                    worksheet1.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})

                    # Apply column-specific formats
                    worksheet1.set_default_row(22)
                    worksheet1.set_column('A:A', 12, None)  # date
                    worksheet1.set_column('B:B', 12, None)  # time
                    worksheet1.set_column('C:C', 12, None)  # timezone
                    worksheet1.set_column('G:G', 12, None)  # author
                    worksheet1.set_column('H:H', 40, None)  # headline
                    worksheet1.set_column('Y:Y', 12, number_format)  # impressions
                    worksheet1.set_column('X:X', 12, currency_format)  # AVE
                    worksheet1.set_column('AH:AH', 12, None)  # Assigned Sentiment
                    worksheet1.set_column('AI:AI', 12, None)  # Flagged
                    worksheet1.set_column('AJ:AJ', 12, None)  # Group ID
                    worksheet1.freeze_panes(1, 0)

                    # Add another worksheet with st.session_state.full_dataset called 'RAW DATA'
                    raw_data = st.session_state.full_dataset
                    raw_data.to_excel(writer, sheet_name='RAW DATA', startrow=1, header=False, index=False)
                    worksheet2 = writer.sheets['RAW DATA']
                    # worksheet2.set_tab_color('grey')

                    (max_row, max_col) = raw_data.shape
                    column_settings = [{'header': column} for column in raw_data.columns]
                    worksheet2.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})


                    # Apply column-specific formats to the 'RAW DATA' worksheet
                    # (Modify this part based on the structure of your raw data)
                    worksheet2.set_default_row(22)
                    worksheet2.set_column('A:A', 12, None)  # date
                    worksheet2.set_column('B:B', 12, None)  # time
                    worksheet2.set_column('C:C', 10, None)  # timezone
                    worksheet2.set_column('G:G', 12, None)  # author
                    worksheet2.set_column('H:H', 40, None)  # headline
                    worksheet2.set_column('Y:Y', 12, number_format)  # impressions
                    worksheet2.set_column('X:X', 12, currency_format)  # AVE
                    worksheet2.freeze_panes(1, 0)


                workbook.close()
                output.seek(0)  # Important: move back to the beginning of the BytesIO object

                # Update the session state variable with the download data
                st.session_state.download_data = output

    # Check if the download data is ready and display the download button
    if st.session_state.download_data is not None:
        export_name = f"{st.session_state.client_name} - {st.session_state.focus} - Sentiment.xlsx"
        st.download_button('Download', st.session_state.download_data, file_name=export_name, type="primary")