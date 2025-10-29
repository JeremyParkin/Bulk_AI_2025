import math
import re
import string
import time
from typing import List

import pandas as pd
import streamlit as st
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

import mig_functions as mig


# Set Streamlit configuration
st.set_page_config(
    page_title="MIG Bulk AI Analysis",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide",
)

# Set the current page in session state
st.session_state.current_page = "Configuration"

# Sidebar configuration
mig.standard_sidebar()

# Initialize st.session_state.elapsed_time if it does not exist
if "elapsed_time" not in st.session_state:
    st.session_state.elapsed_time = 0


def normalize_text(text: str) -> str:
    """Convert to lowercase, remove extra spaces, remove punctuation, etc."""
    text = str(text)
    text = text.lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def remove_extra_spaces(text: str) -> str:
    """Remove extra spaces from the beginning/end and collapse internal whitespace."""
    text = str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_snippet(snippet: str) -> str:
    """Remove leading broadcast markers from snippets."""
    snippet = str(snippet)
    if snippet.startswith(">>>"):
        return snippet[3:]
    if snippet.startswith(">>"):
        return snippet[2:]
    return snippet


def ensure_published_date(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a Published Date column exists and is string formatted."""
    if "Published Date" in df.columns:
        published = pd.to_datetime(df["Published Date"], errors="coerce")
    elif "Date" in df.columns:
        published = pd.to_datetime(df["Date"], errors="coerce")
    else:
        published = pd.Series(pd.NaT, index=df.index)

    df = df.copy()
    df["Published Date"] = published.dt.strftime("%Y-%m-%d")
    return df


def split_batches_by_date(media_df: pd.DataFrame, max_batch_size: int) -> List[pd.DataFrame]:
    """Split a media slice into contiguous batches without breaking dates when possible.

    If one day's coverage exceeds ``max_batch_size`` we split that day into
    multiple batches while preserving ordering.
    """
    if media_df.empty:
        return []

    ordered = media_df.sort_values(
        ["Published Date", "Headline"],
        na_position="last",
        kind="mergesort",  # Stable sort so uploads retain their relative order.
    ).reset_index(drop=True)

    batches: List[pd.DataFrame] = []
    current_indices: List[int] = []
    current_size = 0

    def flush_current() -> None:
        nonlocal current_indices, current_size
        if current_indices:
            batches.append(ordered.iloc[current_indices].copy())
            current_indices = []
            current_size = 0

    date_keys = ordered["Published Date"].fillna("__UNKNOWN__")

    for key, group in ordered.groupby(date_keys, sort=False):
        published_date = group["Published Date"].iloc[0] if key != "__UNKNOWN__" else None
        group_indices = group.index.to_list()

        if len(group_indices) <= max_batch_size:
            if current_size + len(group_indices) > max_batch_size:
                flush_current()

            current_indices.extend(group_indices)
            current_size += len(group_indices)
            continue

        # Large single-day buckets get chunked into sub-batches of max_batch_size
        flush_current()
        for start in range(0, len(group_indices), max_batch_size):
            chunk = group_indices[start : start + max_batch_size]
            batches.append(ordered.iloc[chunk].copy())

        date_label = published_date if published_date is not None else "Unknown date"
        st.info(
            f"Split {date_label} coverage into "
            f"{math.ceil(len(group_indices) / max_batch_size)} batches to stay within resource limits."
        )

    flush_current()
    return batches


def cluster_similar_stories(df: pd.DataFrame, similarity_threshold: float) -> pd.DataFrame:
    """Cluster similar stories using sparse cosine similarity and connected components."""
    df = df.copy()

    texts = (df["Normalized Headline"] + " " + df["Normalized Snippet"]).fillna("")
    if not any(text.strip() for text in texts):
        df["Group ID"] = range(len(df))
        return df

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    if tfidf_matrix.shape[0] == 1:
        df["Group ID"] = 0
        return df

    # Build a sparse similarity graph within the chosen radius
    radius = max(1.0 - similarity_threshold, 0.0)
    nn = NearestNeighbors(metric="cosine", radius=radius)
    nn.fit(tfidf_matrix)
    graph = nn.radius_neighbors_graph(tfidf_matrix, radius=radius, mode="connectivity")

    # Ensure every node is part of its own component at minimum
    graph = graph + sparse.eye(graph.shape[0], format="csr")

    _, labels = csgraph.connected_components(graph, directed=False)
    df["Group ID"] = labels
    return df


def cluster_by_media_type(
    df: pd.DataFrame, similarity_threshold: float, max_batch_size: int = 1800
) -> pd.DataFrame:
    """Cluster stories by media type with date-aware batching and unique IDs."""
    type_column = "Media Type" if "Media Type" in df.columns else "Type"
    clustered_frames: List[pd.DataFrame] = []
    group_id_offset = 0

    for media_type in df[type_column].dropna().unique():
        st.write(f"Processing media type: {media_type}")
        media_df = df[df[type_column] == media_type].copy()

        if media_df.empty:
            continue

        media_df["Headline"] = media_df["Headline"].fillna("").apply(remove_extra_spaces)
        media_df["Snippet"] = media_df["Snippet"].fillna("").apply(remove_extra_spaces).apply(clean_snippet)
        media_df = ensure_published_date(media_df)
        media_df["Normalized Headline"] = media_df["Headline"].apply(normalize_text)
        media_df["Normalized Snippet"] = media_df["Snippet"].apply(normalize_text)

        if media_df[["Headline", "Snippet"]].apply(lambda x: x.str.strip()).eq("").all(axis=None):
            st.warning(f"Skipping media type {media_type} due to missing headlines and snippets.")
            continue

        batches = split_batches_by_date(media_df, max_batch_size=max_batch_size)

        for batch in batches:
            if len(batch) == 1:
                batch["Group ID"] = group_id_offset
                group_id_offset += 1
            else:
                clustered_batch = cluster_similar_stories(batch, similarity_threshold)
                clustered_batch["Group ID"] += group_id_offset
                group_id_offset += clustered_batch["Group ID"].max() + 1
                batch = clustered_batch

            batch = batch.drop(columns=["Normalized Headline", "Normalized Snippet"], errors="ignore")
            clustered_frames.append(batch)

    if not clustered_frames:
        return df

    return pd.concat(clustered_frames, ignore_index=True)


# Main title of the page
st.title("Configuration")

# Check if the upload step is completed
if not st.session_state.upload_step:
    st.error("Please upload a CSV/XLSX before trying this step.")
else:
    if not st.session_state.config_step:
        named_entity = st.session_state.client_name

        # Sampling options
        sampling_option = st.radio(
            "Sampling options:",
            [
                "Take a statistically significant sample",
                "Set my own sample size",
                "Use full data",
            ],
            help="Choose how to sample your uploaded data set.",
        )

        if sampling_option == "Take a statistically significant sample":

            def calculate_sample_size(
                population_size: int,
                confidence_level: float = 0.95,
                margin_of_error: float = 0.05,
                p: float = 0.5,
            ) -> int:
                z_score = 1.96  # For 95% confidence
                numerator = population_size * (z_score**2) * p * (1 - p)
                denominator = (margin_of_error**2) * (population_size - 1) + (z_score**2) * p * (1 - p)
                return math.ceil(numerator / denominator)

            population_size = len(st.session_state.full_dataset)
            st.session_state.sample_size = calculate_sample_size(population_size)
            st.write(f"Calculated sample size: {st.session_state.sample_size}")

        elif sampling_option == "Set my own sample size":
            max_sample = len(st.session_state.full_dataset)
            custom_sample_size = st.number_input(
                "Enter your desired sample size:",
                min_value=1,
                max_value=max_sample,
                step=1,
                value=min(400, max_sample),
            )
            st.session_state.sample_size = int(custom_sample_size)

        else:
            st.session_state.sample_size = len(st.session_state.full_dataset)
            st.write(f"Full data size: {st.session_state.sample_size}")

        similarity_threshold = 0.93
        st.session_state.similarity_threshold = similarity_threshold

        if st.button("Save Configuration", type="primary"):
            start_time = time.time()
            st.session_state.config_step = True

            sample_size = st.session_state.sample_size
            if sample_size < len(st.session_state.full_dataset):
                df = (
                    st.session_state.df_traditional.sample(n=sample_size, random_state=1).reset_index(drop=True)
                )
            else:
                df = st.session_state.df_traditional.copy()

            st.write(f"Full data size: {len(st.session_state.df_traditional)}")
            st.write(f"Sample size used: {len(df)}")

            if "Coverage Snippet" in df.columns:
                df = df.rename(columns={"Coverage Snippet": "Snippet"})

            df = cluster_by_media_type(df, similarity_threshold=similarity_threshold, max_batch_size=1800)

            st.session_state.df_traditional = df.copy()

            group_counts = df.groupby("Group ID").size().reset_index(name="Group Count")
            unique_stories = df.groupby("Group ID").agg(lambda x: x.iloc[0]).reset_index()
            unique_stories_with_counts = unique_stories.merge(group_counts, on="Group ID")
            unique_stories_sorted = unique_stories_with_counts.sort_values(
                by="Group Count", ascending=False
            ).reset_index(drop=True)
            st.session_state.unique_stories = unique_stories_sorted

            end_time = time.time()
            st.session_state.elapsed_time = end_time - start_time
            st.rerun()

    else:
        st.success("Configuration Completed!")
        st.write(f"Time taken: {st.session_state.elapsed_time:.2f} seconds")
        st.write(f"Full data size: {len(st.session_state.full_dataset)}")
        if "sample_size" in st.session_state:
            st.write(f"Sample size used: {st.session_state.sample_size}")
        st.write(f"Unique stories in data: {len(st.session_state.unique_stories)}")
        st.dataframe(st.session_state.unique_stories)

        def reset_config() -> None:
            """Reset the configuration step and related session state variables."""
            st.session_state.config_step = False
            st.session_state.sentiment_opinion = None
            st.session_state.random_sample = None
            st.session_state.similarity_threshold = None
            st.session_state.sentiment_instruction = None
            st.session_state.df_traditional = st.session_state.full_dataset.copy()
            st.session_state.counter = 0
            st.session_state.pop("unique_stories", None)

        if st.button("Reset Configuration"):
            reset_config()
            st.rerun()
