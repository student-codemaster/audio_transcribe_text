"""Enhanced Streamlit app with comprehensive visualizations."""

import streamlit as st
import os
import shutil
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import json

# Configure network timeouts and caching
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow online model loading
os.environ['HF_HUB_OFFLINE'] = '0'

# Use lazy imports to handle network timeouts gracefully
try:
    from pipeline import run_pipeline
    from visualization import (
        create_segment_timeline,
        create_sentiment_trend,
        create_keyword_cloud,
        create_keyword_bar_chart,
        create_segment_distribution,
        create_sentiment_heatmap,
        create_keywords_per_segment
    )
    IMPORTS_SUCCESSFUL = True
except Exception as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)

# Page config
st.set_page_config(
    page_title="AI Audio Transcriber", 
    layout="wide",
    page_icon="🎙️",
    initial_sidebar_state="expanded"
)

st.title(" AI Audio Transcriber & Analyzer")
st.markdown("---")

# Check if imports were successful
if not IMPORTS_SUCCESSFUL:
    st.error(" Failed to initialize application")
    st.error(f"Error: {IMPORT_ERROR}")
    st.info("""
    This usually means:
    1. Network timeout downloading ML models
    2. Missing dependencies
    
    **Solutions:**
    1. Check your internet connection
    2. Restart the app: `streamlit run streamlit_app_v2.py`
    3. Reinstall dependencies: `pip install -r requirements.txt`
    """)
    st.stop()

# Initialize session state
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.index = None
    st.session_state.search_engine = None
    st.session_state.topics = None
    st.session_state.transcript = None
    st.session_state.sentiment_score = None

# Sidebar
with st.sidebar:
    st.header(" Configuration")
    st.markdown("Upload your podcast audio file and let AI analyze it!")
    
    uploaded_file = st.file_uploader(" Upload Podcast Audio", type=["wav", "mp3"], 
                                   help="Supported formats: WAV, MP3")
    
    if uploaded_file and st.button(" Analyze Podcast", use_container_width=True, type="primary"):
        with st.spinner("Processing audio... This may take a few minutes"):
            # Save uploaded file temporarily then copy to data folder
            local_path = uploaded_file.name
            with open(local_path, "wb") as f:
                f.write(uploaded_file.read())

            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)
            stored_path = os.path.join(data_dir, os.path.basename(local_path))
            # ensure unique name if file already exists
            counter = 1
            base, ext = os.path.splitext(stored_path)
            while os.path.exists(stored_path):
                stored_path = f"{base}_{counter}{ext}"
                counter += 1
            shutil.copy(local_path, stored_path)

            try:
                # Run pipeline on stored audio
                transcript, topics, sentiment_score, summaries, index, search_engine = run_pipeline(stored_path)
                
                # Store in session state
                st.session_state.transcript = transcript
                st.session_state.topics = topics
                st.session_state.sentiment_score = sentiment_score
                st.session_state.index = index
                st.session_state.search_engine = search_engine
                st.session_state.analysis_complete = True
                st.session_state.audio_path = stored_path
                
                st.success(" Analysis complete! Explore visualizations below.")
                
            except Exception as e:
                error_msg = str(e)
                st.error(f" Analysis failed: {error_msg}")
                
                if "timeout" in error_msg.lower() or "read" in error_msg.lower():
                    st.warning("""
                    **Network Timeout:** The model downloading took too long.
                    
                    Solutions:
                    1. Check your internet connection
                    2. Try uploading a smaller file first
                    3. Run troubleshoot.py: `python troubleshoot.py`
                    4. Retry in a few moments
                    """)
                else:
                    st.warning("Check the audio file format and try again.")

# Main content areas
if not st.session_state.analysis_complete:
    st.info(" Upload an audio file and click ' Analyze Podcast' to begin")
    st.markdown("""
    ###  What This App Does:
    1. **Transcription**: Converts audio to text using AI
    2. **Segmentation**: Breaks down content into logical segments
    3. **Analysis**: Extracts keywords, sentiments, and summaries
    4. **Visualization**: Creates interactive charts and dashboards
    """)
else:
    index = st.session_state.index
    search_engine = st.session_state.search_engine
    topics = st.session_state.topics
    transcript = st.session_state.transcript
    sentiment_score = st.session_state.sentiment_score
    
    # Display overall metrics
    st.subheader(" Episode Overview")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric(" Total Segments", len(topics))
    with metric_col2:
        sentiment_icon = "" if sentiment_score > 0.5 else "" if sentiment_score < -0.5 else "😐"
        st.metric(f"{sentiment_icon} Avg Sentiment", f"{sentiment_score:.3f}")
    with metric_col3:
        st.metric(" Total Duration", f"{index.segments[-1]['end']:.1f}s")
    with metric_col4:
        st.metric(" Word Count", len(transcript.split()))
    
    # provide download options
    if st.session_state.get("audio_path"):
        audio_name = os.path.basename(st.session_state.audio_path)
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            try:
                with open(st.session_state.audio_path, "rb") as af:
                    st.download_button(" Download Audio", data=af.read(), file_name=audio_name)
            except Exception:
                pass
        
        with download_col2:
            episode_id = os.path.splitext(audio_name)[0]
            segment_json = os.path.join("final_outputs", f"{episode_id}.json")
            if os.path.exists(segment_json):
                with open(segment_json, "rb") as jf:
                    st.download_button(" Download Segments JSON", data=jf.read(), 
                                     file_name=f"{episode_id}.json", mime="application/json")
    
    st.divider()
    
    # Sidebar navigation
    with st.sidebar:
        st.header(" Navigation")
        page = st.radio(
            "Choose a view:",
            [" Transcript", " Search", " Analytics", " Timeline", " Keywords", " Sentiment", " Multi-Episode Test"],
            index=0,
            help="Select a view to explore your podcast analysis"
        )
    
    # ============================================
    # PAGE: Transcript
    # ============================================
    if page == " Transcript":
        st.header(" Transcripts & Segments")
        
        # Tabs for transcript vs segments
        inner_tab1, inner_tab2 = st.tabs(["Full Transcript", "Segments View"])
        
        with inner_tab1:
            st.subheader(" Complete Episode Transcript")
            if transcript:
                st.write(transcript)
            else:
                st.info("No transcript available.")
        
        with inner_tab2:
            # Add controls for segments view
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            with col1:
                show_summaries = st.checkbox("Show Summaries", value=True)
            with col2:
                show_keywords = st.checkbox("Show Keywords", value=True)
            with col3:
                compact_view = st.checkbox("Compact View", value=False)
            with col4:
                show_raw = st.checkbox("Show Raw Data", value=False)
            
            # Create a scrollable container
            with st.container(height=600):
                for i, seg in enumerate(index.segments):
                    # Segment header with styling
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0;">
                        <strong>Segment {seg['id']}</strong> | 
                        <span style="color: #0066cc;">[{index._format_time(seg['start'])} - {index._format_time(seg['end'])}]</span> 
                        <span style="color: #666;">({seg['duration']:.1f}s)</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if show_raw:
                        st.code(str(seg), language='json')
                    
                    if show_summaries and seg.get('summary'):
                        st.markdown(f"**📝 Summary:** {seg['summary']}")
                    
                    text_content = seg.get("segments", seg.get("text", ""))
                    if compact_view:
                        preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
                        st.write(preview)
                        if len(text_content) > 200:
                            with st.expander("Read full text"):
                                st.write(text_content)
                    else:
                        st.write(text_content)
                    
                    if show_keywords and seg.get('keywords'):
                        st.markdown(f"**🏷️ Keywords:** {', '.join(seg['keywords'])}")
                    
                    sentiment = seg.get('sentiment_score', 0)
                    if sentiment > 0.5:
                        st.success(f"😊 Positive ({sentiment:.2f})")
                    elif sentiment < -0.5:
                        st.error(f"😞 Negative ({sentiment:.2f})")
                    else:
                        st.info(f"😐 Neutral ({sentiment:.2f})")
                    
                    st.divider()
    
    # ============================================
    # PAGE: Timeline
    # ============================================
    elif page == "⏱️ Timeline":
        st.header("⏱️ Interactive Segment Timeline")
        
        # Timeline display options
        timeline_opt1, timeline_opt2, timeline_opt3 = st.columns(3)
        with timeline_opt1:
            show_summary = st.checkbox("Show Summaries", value=True)
        with timeline_opt2:
            show_keywords = st.checkbox("Show Keywords", value=True)
        with timeline_opt3:
            show_details = st.checkbox("Show Sentiment", value=True)
        
        st.divider()
        
        # Main timeline visualization
        with st.spinner("🎬 Generating timeline visualization..."):
            fig_timeline = create_segment_timeline(index.segments)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.divider()
        
        # Detailed timeline view
        st.subheader("📍 Timeline Details")
        
        for i, seg in enumerate(index.segments):
            col1, col2, col3 = st.columns([1, 4, 1])
            
            with col1:
                st.metric(f"Segment {seg['id']}", f"{seg['duration']:.1f}s")
            
            with col2:
                st.write(f"**⏰ {index._format_time(seg['start'])} → {index._format_time(seg['end'])}**")
                if show_summary and seg.get('summary'):
                    st.caption(f"📝 {seg['summary']}")
                if show_keywords and seg.get('keywords'):
                    st.caption(f"🏷️ {', '.join(seg['keywords'])}")
            
            with col3:
                if show_details:
                    sentiment_emoji = "😊" if seg['sentiment_score'] > 0.5 else "😞" if seg['sentiment_score'] < -0.5 else "😐"
                    st.metric(sentiment_emoji, f"{seg['sentiment_score']:.2f}")
            
            st.divider()
    
    # ============================================
    # PAGE: Keywords & Topics
    # ============================================
    elif page == "🏷️ Keywords":
        st.header("🏷️ Keyword Analysis & Topics")
        
        # Collect all keywords
        all_keywords = []
        keyword_freq = {}
        keyword_segments = {}
        
        for seg in index.segments:
            for kw in seg['keywords']:
                all_keywords.append(kw)
                keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
                if kw not in keyword_segments:
                    keyword_segments[kw] = []
                keyword_segments[kw].append(seg['id'])
        
        # Controls
        kw_col1, kw_col2 = st.columns([1, 3])
        with kw_col1:
            min_freq = st.slider("Min Frequency", 1, max(keyword_freq.values()) if keyword_freq else 1, 1)
        with kw_col2:
            selected_kw = st.multiselect("Filter Keywords", 
                                       sorted(keyword_freq.keys()), 
                                       default=sorted(keyword_freq.keys())[:5] if len(keyword_freq) > 5 else sorted(keyword_freq.keys()))
        
        # Filter keywords
        filtered_freq = {k: v for k, v in keyword_freq.items() if v >= min_freq and k in selected_kw}
        
        st.divider()
        
        # Row 1: Word Cloud and Bar Chart
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🌈 Word Cloud")
            if filtered_freq:
                fig_wc = create_keyword_cloud(index.segments, num_keywords=30)
                st.pyplot(fig_wc)
            else:
                st.info("No keywords match the current filters.")
        
        with col2:
            st.subheader("📊 Top Keywords Bar Chart")
            if filtered_freq:
                fig_kw_bar = create_keyword_bar_chart(index.segments, num_keywords=15)
                st.plotly_chart(fig_kw_bar, use_container_width=True)
        
        st.divider()
        
        # Keywords per segment table
        st.subheader("📋 Keywords by Segment")
        df_seg_kw = []
        for seg in index.segments:
            df_seg_kw.append({
                "Segment": f"S{seg['id']}",
                "Keywords": ', '.join(seg['keywords']),
                "Count": len(seg['keywords']),
                "Duration": f"{seg['duration']:.1f}s",
                "Sentiment": f"{seg['sentiment_score']:.2f}"
            })
        
        df_seg_kw = pd.DataFrame(df_seg_kw)
        st.dataframe(df_seg_kw, use_container_width=True, 
                   column_config={
                       "Keywords": st.column_config.TextColumn("Keywords", width="large"),
                       "Count": st.column_config.NumberColumn("Count", format="%d"),
                       "Duration": st.column_config.TextColumn("Duration"),
                       "Sentiment": st.column_config.NumberColumn("Sentiment", format="%.2f")
                   })
        
        # Keyword co-occurrence
        if len(filtered_freq) > 1:
            st.divider()
            st.subheader("🔗 Keyword Co-occurrence")
            kw_list = list(filtered_freq.keys())[:10]
            co_occurrence = {}
            
            for seg in index.segments:
                seg_kws = set(seg['keywords']) & set(kw_list)
                for kw1 in seg_kws:
                    for kw2 in seg_kws:
                        if kw1 != kw2:
                            key = tuple(sorted([kw1, kw2]))
                            co_occurrence[key] = co_occurrence.get(key, 0) + 1
            
            if co_occurrence:
                top_co = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:10]
                st.write("**Top keyword pairs:**")
                for pair, count in top_co:
                    st.write(f"• {pair[0]} ↔ {pair[1]}: appears together in {count} segments")
    
    # ============================================
    # PAGE: Sentiment Analysis
    # ============================================
    elif page == "😊 Sentiment":
        st.header("😊 Sentiment Analysis & Emotional Trends")
        
        df_sentiment = []
        for seg in index.segments:
            df_sentiment.append({
                "Segment": f"S{seg['id']}",
                "Sentiment": seg['sentiment_score'],
                "Timestamp": f"{index._format_time(seg['start'])}",
                "Duration": seg['duration'],
                "Word Count": len(seg.get("segments", seg.get("text", "")).split())
            })
        
        df_sentiment = pd.DataFrame(df_sentiment)
        
        # Sentiment controls
        sent_col1, sent_col2, sent_col3 = st.columns([1, 1, 2])
        with sent_col1:
            show_trend = st.checkbox("Show Trend Line", value=True)
        with sent_col2:
            show_zones = st.checkbox("Show Sentiment Zones", value=True)
        with sent_col3:
            sentiment_view = st.radio("View Type", ["Line Chart", "Bar Chart", "Area Chart"], horizontal=True)
        
        st.divider()
        
        # Main sentiment trend
        st.subheader("📈 Sentiment Trend Over Time")
        fig_sentiment = create_sentiment_trend(index.segments)
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        st.divider()
        
        # Row 1: Heatmap and Distribution
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔥 Sentiment Heatmap")
            fig_heatmap = create_sentiment_heatmap(index.segments)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            st.subheader("📊 Sentiment Distribution")
            fig_dist = px.box(df_sentiment, y="Sentiment", 
                            title="Sentiment Distribution",
                            color_discrete_sequence=["#FF6B6B"])
            fig_dist.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        st.divider()
        
        # Sentiment statistics
        st.subheader("📊 Sentiment Statistics")
        stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
        
        with stat_col1:
            st.metric("Avg Sentiment", f"{df_sentiment['Sentiment'].mean():.2f}")
        with stat_col2:
            st.metric("Max Sentiment", f"{df_sentiment['Sentiment'].max():.2f}")
        with stat_col3:
            st.metric("Min Sentiment", f"{df_sentiment['Sentiment'].min():.2f}")
        with stat_col4:
            st.metric("Std Dev", f"{df_sentiment['Sentiment'].std():.2f}")
        with stat_col5:
            positive_count = len(df_sentiment[df_sentiment['Sentiment'] > 0.5])
            st.metric("Positive Segments", positive_count)
        
        st.divider()
        
        # Sentiment vs other metrics
        st.subheader("🔗 Sentiment Correlations")
        
        corr_col1, corr_col2 = st.columns(2)
        
        with corr_col1:
            fig_corr1 = px.scatter(df_sentiment, x="Duration", y="Sentiment", 
                                 title="Sentiment vs Duration",
                                 color="Sentiment", size="Word Count",
                                 color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_corr1, use_container_width=True)
        
        with corr_col2:
            fig_corr2 = px.scatter(df_sentiment, x="Word Count", y="Sentiment", 
                                 title="Sentiment vs Word Count",
                                 color="Sentiment", size="Duration",
                                 color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_corr2, use_container_width=True)
    
    # ============================================
    # PAGE: Analytics Dashboard
    # ============================================
    elif page == "📊 Analytics":
        st.header("Comprehensive Analytics Dashboard")
        
        # Create analytics DataFrame
        seg_data = []
        for seg in index.segments:
            seg_data.append({
                "Segment": f"S{seg['id']}",
                "Start": seg['start'],
                "End": seg['end'],
                "Duration": seg['duration'],
                "Sentiment": seg['sentiment_score'],
                "Keyword Count": len(seg['keywords']),
                "Word Count": len(seg.get("segments", seg.get("text", "")).split())
            })
        
        df = pd.DataFrame(seg_data)
        
        # Key metrics row
        st.subheader("📌 Key Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
        with metric_col1:
            st.metric("Total Segments", len(df))
        with metric_col2:
            st.metric("Avg Duration", f"{df['Duration'].mean():.1f}s")
        with metric_col3:
            st.metric("Max Duration", f"{df['Duration'].max():.1f}s")
        with metric_col4:
            st.metric("Avg Sentiment", f"{df['Sentiment'].mean():.2f}")
        with metric_col5:
            st.metric("Sentiment Std Dev", f"{df['Sentiment'].std():.2f}")
        
        st.divider()
        
        # Charts row 1
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("⏱️ Segment Duration Distribution")
            fig = px.bar(df, x="Segment", y="Duration", 
                        title="Duration per Segment",
                        color="Duration",
                        color_continuous_scale="Blues")
            fig.update_layout(xaxis_title='Segment', yaxis_title='Duration (s)')
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_col2:
            st.subheader("😊 Sentiment Score Distribution")
            fig = px.box(df, y="Sentiment", 
                        title="Sentiment Distribution",
                        color_discrete_sequence=["#FF6B6B"])
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Charts row 2
        chart_col3, chart_col4 = st.columns(2)
        
        with chart_col3:
            st.subheader("📝 Word Count per Segment")
            fig = px.line(df, x="Segment", y="Word Count", 
                         markers=True, title="Word Count Progression",
                         color_discrete_sequence=["#4ECDC4"])
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_col4:
            st.subheader("🏷️ Keywords per Segment")
            fig = px.bar(df, x="Segment", y="Keyword Count", 
                        title="Keyword Count Distribution",
                        color="Keyword Count",
                        color_continuous_scale="Greens")
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Additional visualizations
        st.subheader("🎯 Advanced Analysis")
        
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            st.subheader("📊 Segment Distribution (Pie)")
            fig_dist = create_segment_distribution(index.segments)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with adv_col2:
            st.subheader("🔗 Keywords vs Sentiment")
            fig_corr = px.scatter(df, x="Keyword Count", y="Sentiment", 
                                size="Duration", color="Sentiment",
                                title="Keywords vs Sentiment",
                                color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_corr, use_container_width=True)
        
        st.divider()
        
        # Detailed table
        st.subheader("📋 Detailed Segment Data")
        st.dataframe(df, use_container_width=True,
                   column_config={
                       "Segment": st.column_config.TextColumn("Segment"),
                       "Duration": st.column_config.NumberColumn("Duration (s)", format="%.1f"),
                       "Sentiment": st.column_config.NumberColumn("Sentiment", format="%.2f"),
                       "Keyword Count": st.column_config.NumberColumn("Keywords", format="%d"),
                       "Word Count": st.column_config.NumberColumn("Words", format="%d")
                   })
    
    # ============================================
    # PAGE: Search
    # ============================================
    elif page == "🔍 Search":
        st.header("🔍 Search & Query Analysis")
        
        if search_engine is None:
            st.warning("⚠️ No transcript loaded. Please upload an audio file first.")
        else:
            st.markdown("""
            Search through your transcript using three methods:
            - **Keyword Search**: Find segments by matching text
            - **Semantic Search**: Find semantically similar segments
            - **Combined**: Hybrid approach using both methods
            """)
            
            # Search configuration
            search_col1, search_col2, search_col3 = st.columns(3)
            
            with search_col1:
                search_query = st.text_input(
                    "🔎 Enter search query:",
                    placeholder="e.g., 'machine learning', 'data science'",
                    key="search_query"
                )
            
            with search_col2:
                search_type = st.radio(
                    "Search Method:",
                    ["Keyword", "Semantic", "Combined"],
                    horizontal=True
                )
            
            with search_col3:
                num_results = st.slider(
                    "Results to show:",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1
                )
            
            # Advanced options
            st.markdown("---")
            with st.expander("⚙️ Advanced Search Options"):
                adv_col1, adv_col2, adv_col3 = st.columns(3)
                
                with adv_col1:
                    if search_type == "Semantic":
                        threshold = st.slider(
                            "Similarity threshold:",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.3,
                            step=0.05,
                            help="Minimum similarity score to include results"
                        )
                    else:
                        threshold = 0.3
                
                with adv_col2:
                    if search_type == "Combined":
                        keyword_weight = st.slider(
                            "Keyword weight:",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.4,
                            step=0.1
                        )
                    else:
                        keyword_weight = 0.4
                
                with adv_col3:
                    if search_type == "Combined":
                        semantic_weight = st.slider(
                            "Semantic weight:",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.6,
                            step=0.1
                        )
                    else:
                        semantic_weight = 0.6
            
            # Perform search
            if search_query and search_query.strip():
                st.markdown("---")
                st.subheader(f"📊 Search Results for: \"{search_query}\"")
                
                try:
                    if search_type == "Keyword":
                        results = search_engine.keyword_search(search_query, top_k=num_results)
                        result_type = "Keyword Match"
                    elif search_type == "Semantic":
                        results = search_engine.semantic_search(
                            search_query, 
                            top_k=num_results,
                            threshold=threshold
                        )
                        result_type = "Semantic Similarity"
                    else:  # Combined
                        results = search_engine.combined_search(
                            search_query,
                            top_k=num_results,
                            keyword_weight=keyword_weight,
                            semantic_weight=semantic_weight
                        )
                        result_type = "Combined Score"
                    
                    if results:
                        # Display results count
                        if search_type == "Keyword":
                            st.success(f" Found {len(results)} matching segment(s)")
                        else:
                            st.success(f" Found {len(results)} result(s)")
                        
                        # Display each result
                        for idx, (result, score) in enumerate(results, 1):
                            result_container = st.container(border=True)
                            
                            with result_container:
                                # Header with segment info
                                header_cols = st.columns([1, 3, 2])
                                
                                with header_cols[0]:
                                    st.metric("Segment ID", result.get("id", "N/A"))
                                
                                with header_cols[1]:
                                    start_time = result.get("start", 0)
                                    end_time = result.get("end", 0)
                                    duration = result.get("duration", 0)
                                    st.metric("Time Range", f"{start_time:.1f}s - {end_time:.1f}s")
                                
                                with header_cols[2]:
                                    if search_type == "Semantic":
                                        st.metric(f"{result_type}", f"{score:.2%}")
                                    elif search_type == "Combined":
                                        st.metric(f"{result_type}", f"{score:.3f}")
                                    else:
                                        st.metric("Match Type", "Keyword")
                                
                                # Segment text
                                st.markdown("** Segment Text:**")
                                segment_text = result.get("segments", result.get("text", ""))
                                
                                # Highlight keyword if found
                                if search_type == "Keyword":
                                    query_lower = search_query.lower()
                                    text_lower = segment_text.lower()
                                    if query_lower in text_lower:
                                        # Simple highlighting by wrapping in markers
                                        highlighted = segment_text.replace(
                                            search_query,
                                            f"**{search_query}**"
                                        )
                                        st.markdown(highlighted)
                                    else:
                                        st.write(segment_text)
                                else:
                                    st.write(segment_text)
                                
                                # Metadata row
                                meta_cols = st.columns(3)
                                
                                with meta_cols[0]:
                                    keywords = result.get("keywords", [])
                                    if keywords:
                                        st.write(f"**Keywords:** {', '.join(keywords[:5])}")
                                
                                with meta_cols[1]:
                                    summary = result.get("summary", "N/A")
                                    if summary and summary != "N/A":
                                        st.write(f"**Summary:** {summary[:100]}...")
                                
                                with meta_cols[2]:
                                    sentiment = result.get("sentiment_score", 0)
                                    sentiment_label = "😊 Positive" if sentiment > 0.1 else ("😞 Negative" if sentiment < -0.1 else "😐 Neutral")
                                    st.write(f"**Sentiment:** {sentiment_label} ({sentiment:.2f})")
                    else:
                        st.info(f"ℹ️ No results found for '{search_query}' using {search_type} search")
                        
                        if search_type == "Semantic" and threshold > 0.5:
                            st.tip("Try lowering the similarity threshold to get more results")
                
                except Exception as e:
                    st.error(f"❌ Search error: {str(e)}")
                    if "embedding" in str(e).lower() or "model" in str(e).lower():
                        st.info("💡 Semantic search requires embeddings. Try Keyword search instead.")
            else:
                st.info("📝 Enter a search query to get started")
    
    # ============================================
    # PAGE: Multi-Episode Testing
    # ============================================
    elif page == "🎯 Multi-Episode Test":
        st.header("🎯 Multi-Episode Testing & Comparison")
        
        st.info("📌 This section is for testing the pipeline on multiple podcast episodes.")
        
        # Create test report
        st.subheader("📊 Current Episode Analysis Report")
        
        report_col1, report_col2 = st.columns(2)
        
        with report_col1:
            st.write("**Segmentation Quality Assessment**")
            st.write(f"✓ Total Segments: {len(index.segments)}")
            st.write(f"✓ Avg Segment Duration: {np.mean([s['duration'] for s in index.segments]):.1f}s")
            st.write(f"✓ Duration Range: {min([s['duration'] for s in index.segments]):.1f}s - {max([s['duration'] for s in index.segments]):.1f}s")
        
        with report_col2:
            st.write("**Keyword & Topic Analysis**")
            all_kws = [kw for seg in index.segments for kw in seg['keywords']]
            st.write(f"✓ Total Keywords Extracted: {len(all_kws)}")
            st.write(f"✓ Unique Keywords: {len(set(all_kws))}")
            st.write(f"✓ Avg Keywords per Segment: {len(all_kws) / len(index.segments):.1f}")
        
        st.divider()
        
        st.write("**Summary Quality Assessment**")
        summary_quality = []
        for i, seg in enumerate(index.segments):
            summary = seg.get('summary', '')
            quality = len(summary.split()) / max(len(seg.get("segments", "").split()), 1)
            quality_score = min(quality, 1.0)
            summary_quality.append({
                'Segment': f"S{seg['id']}",
                'Summary': summary[:100] + '...' if len(summary) > 100 else summary,
                'Quality': quality_score
            })
        
        df_quality = pd.DataFrame(summary_quality)
        
        fig_quality = px.bar(df_quality, x='Segment', y='Quality',
                            title='Summary Quality Score per Segment',
                            color='Quality',
                            color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_quality, use_container_width=True)
        
        st.divider()
        
        st.subheader("💾 Export Test Results")
        
        # Prepare comprehensive report
        report_data = {
            'Episode Analysis Report': {
                'Total Segments': len(index.segments),
                'Avg Sentiment': float(sentiment_score),
                'Total Duration': float(index.segments[-1]['end']),
                'Word Count': len(transcript.split()),
                'Unique Keywords': len(set(all_kws)),
                'Segments': [
                    {
                        'id': seg['id'],
                        'duration': float(seg['duration']),
                        'sentiment': float(seg['sentiment_score']),
                        'keywords_count': len(seg['keywords']),
                        'summary': seg.get('summary', '')
                    }
                    for seg in index.segments
                ]
            }
        }
        
        report_json = json.dumps(report_data, indent=2)
        st.download_button(
            label="📥 Download Test Report (JSON)",
            data=report_json,
            file_name="test_report.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
### 🎙️ AI Audio Transcriber
""")
