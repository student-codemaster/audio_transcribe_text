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

# Page config
st.set_page_config(
    page_title=" AI Audio Transcriber", 
    layout="wide",
    page_icon="🎙️",
    initial_sidebar_state="expanded"
)
st.title("🎙️ AI Audio Transcriber")
st.markdown("---")

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
    st.header("Configuration")
    st.markdown("Upload your podcast audio file and let AI analyze it!")
    
    uploaded_file = st.file_uploader(" Upload Podcast Audio", type=["wav", "mp3"], 
                                   help="Supported formats: WAV, MP3")
    
    if uploaded_file and st.button("Analyze Podcast", use_container_width=True, type="primary"):
        with st.spinner(" Processing audio... This may take a few minutes"):
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
        
        st.success("Analysis complete! You can download the results below.")

# Main content areas
if not st.session_state.analysis_complete:
    st.info(" Upload an audio file and click 'Analyze Podcast' to begin")
else:
    index = st.session_state.index
    search_engine = st.session_state.search_engine
    topics = st.session_state.topics
    transcript = st.session_state.transcript
    sentiment_score = st.session_state.sentiment_score
    
    # Display overall metrics
    st.subheader("📊 Episode Overview")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("🎬 Total Segments", len(topics))
    with metric_col2:
        sentiment_icon = "😊" if sentiment_score > 0.5 else "😞" if sentiment_score < -0.5 else "😐"
        st.metric(f"{sentiment_icon} Avg Sentiment", f"{sentiment_score:.3f}")
    with metric_col3:
        st.metric("⏱️ Duration", f"{index.segments[-1]['end']:.1f}s")
    with metric_col4:
        st.metric("📝 Word Count", len(transcript.split()))
    
    # provide download options for processed audio & json
    if st.session_state.get("audio_path"):
        audio_name = os.path.basename(st.session_state.audio_path)
        try:
            with open(st.session_state.audio_path, "rb") as af:
                st.download_button("Download audio", data=af.read(), file_name=audio_name)
        except Exception:
            pass
        episode_id = os.path.splitext(audio_name)[0]
        segment_json = os.path.join("final_outputs", f"{episode_id}.json")
        if os.path.exists(segment_json):
            with open(segment_json, "rb") as jf:
                st.download_button("Download segments JSON", data=jf.read(), file_name=f"{episode_id}.json", mime="application/json")
    
    st.divider()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio(
            "Choose a view:",
            [" Transcript", "🔍 Search", "📊 Analytics", "⏱️ Timeline", "🏷️ Keywords", "😊 Sentiment"],
            index=0,
            help="Select a view to explore your podcast analysis"
        )
    
    # Page content based on selection
    if page == "📄 Transcript":
        st.header("📄 Transcripts & Segments")
        
        # inner tabs for transcript vs segments
        inner_tab1, inner_tab2 = st.tabs(["Full Transcript", "Segments"])
        
        with inner_tab1:
            st.subheader("Complete Episode Transcript")
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
                    # Segment header with better styling
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0;">
                        <strong>Segment {seg['id']}</strong> | 
                        <span style="color: #0066cc;">[{index._format_time(seg['start'])} - {index._format_time(seg['end'])}]</span> 
                        <span style="color: #666;">({seg['duration']:.1f}s)</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show raw segment data if enabled
                    if show_raw:
                        st.code(str(seg), language='json')
                    
                    # Summary first (if enabled)
                    if show_summaries and seg.get('summary'):
                        st.markdown(f"**📝 Summary:** {seg['summary']}")
                    
                    # Segments text
                    text_content = seg.get("segments", seg.get("text", ""))
                    if compact_view:
                        # Show preview
                        preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
                        st.write(preview)
                        if len(text_content) > 200:
                            with st.expander("Read full text"):
                                st.write(text_content)
                    else:
                        st.write(text_content)
                    
                    # Keywords (if enabled)
                    if show_keywords and seg.get('keywords'):
                        st.markdown(f"**🏷️ Keywords:** {', '.join(seg['keywords'])}")
                    
                    # Sentiment with color coding
                    sentiment = seg.get('sentiment_score', 0)
                    if sentiment > 0.5:
                        st.success(f"😊 Positive ({sentiment:.2f})")
                    elif sentiment < -0.5:
                        st.error(f"😞 Negative ({sentiment:.2f})")
                    else:
                        st.info(f"😐 Neutral ({sentiment:.2f})")
                    
                    st.divider()  # Separator between segments
                    
                    # Jump to time button
                    st.write(f"[Play from {index._format_time(seg['start'])} - {index._format_time(seg['end'])}](#)")
    
    elif page == "🔍 Search":
        st.header("🔍 Smart Segment Search")
        
        # Search controls
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            query = st.text_input("Enter search query (keyword or semantic):", 
                                placeholder="e.g., machine learning, AI trends, podcast topics")
        with col2:
            search_type = st.selectbox("Search Type", ["Combined", "Keyword", "Semantic"])
        with col3:
            top_k = st.selectbox("Results", [5, 10, 15, 20], index=1)
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                min_sentiment = st.slider("Min Sentiment", -1.0, 1.0, -1.0)
            with filter_col2:
                max_sentiment = st.slider("Max Sentiment", -1.0, 1.0, 1.0)
            with filter_col3:
                min_duration = st.slider("Min Duration (s)", 0, 300, 0)
        
        if query:
            with st.spinner("Searching..."):
                if search_type == "Keyword":
                    results = search_engine.keyword_search(query, top_k=top_k)
                    search_results = [(seg, 1.0) for seg in results]
                elif search_type == "Semantic":
                    search_results = search_engine.semantic_search(query, top_k=top_k)
                else:
                    search_results = search_engine.combined_search(query, top_k=top_k)
                
                # Apply filters
                filtered_results = []
                for seg, score in search_results:
                    if (seg['sentiment_score'] >= min_sentiment and 
                        seg['sentiment_score'] <= max_sentiment and 
                        seg['duration'] >= min_duration):
                        filtered_results.append((seg, score))
                
                search_results = filtered_results
            
            st.success(f" Found {len(search_results)} results for '{query}'")
            
            if search_results:
                for i, (seg, score) in enumerate(search_results):
                    # Enhanced expandable container
                    sentiment_color = "🟢" if seg['sentiment_score'] > 0.5 else "🔴" if seg['sentiment_score'] < -0.5 else "🟡"
                    header_text = f"**{i+1}. Segment {seg['id']}** | {index._format_time(seg['start'])} - {index._format_time(seg['end'])} | {sentiment_color} {seg['sentiment_score']:.2f}"
                    
                    with st.expander(header_text, expanded=(i < 3)):  # Expand first 3 results
                        # Content in columns
                        content_col1, content_col2 = st.columns([3, 1])
                        
                        with content_col1:
                            # Full segment text
                            st.subheader(" Full Text")
                            st.write(seg.get("segments", seg.get("text", "")))
                            
                            # Summary
                            if seg.get('summary'):
                                st.subheader("Summary")
                                st.write(seg['summary'])
                            
                            # Keywords
                            if seg.get('keywords'):
                                st.subheader(" Keywords")
                                st.write(", ".join(seg['keywords']))
                        
                        with content_col2:
                            # Metadata
                            st.subheader(" Metadata")
                            st.metric("Duration", f"{seg['duration']:.1f}s")
                            st.metric("Sentiment", f"{seg['sentiment_score']:.2f}")
                            if search_type == "Semantic":
                                st.metric("Similarity", f"{score:.3f}")
                            else:
                                st.metric("Relevance", f"{score:.2f}")
                            st.metric("Word Count", len(seg.get("segments", seg.get("text", "")).split()))
            else:
                st.info("No results found. Try adjusting your search query or filters.")

    
    elif page == "📊 Analytics":
        st.header("Analytics Dashboard")
        
        # Create DataFrame for analytics
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
        st.subheader("Key Metrics")
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
                        title="Sentiment Distribution Across Segments",
                        color_discrete_sequence=["#FF6B6B"])
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)
        
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
        
        # Correlation analysis
        st.subheader("🔗 Correlations")
        corr_col1, corr_col2 = st.columns(2)
        
        with corr_col1:
            # Sentiment vs Duration
            fig = px.scatter(df, x="Duration", y="Sentiment", 
                           title="Sentiment vs Duration Correlation",
                           color="Sentiment", size="Word Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with corr_col2:
            # Word Count vs Keywords
            fig = px.scatter(df, x="Word Count", y="Keyword Count", 
                           title="Word Count vs Keywords Correlation",
                           color="Sentiment", size="Duration")
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "⏱️ Timeline":
        st.header("⏱️ Interactive Timeline")
        
        # Timeline controls
        timeline_col1, timeline_col2 = st.columns([1, 3])
        with timeline_col1:
            show_boundaries = st.checkbox("Show Boundaries", value=True)
        with timeline_col2:
            selected_segment = st.selectbox("Jump to Segment", 
                                          [f"Segment {seg['id']}: {index._format_time(seg['start'])} - {index._format_time(seg['end'])}" 
                                           for seg in index.segments], index=0)
        
        df_timeline = []
        for seg in index.segments:
            df_timeline.append({
                "Segment": f"S{seg['id']}",
                "Start": seg['start'],
                "End": seg['end'],
                "Duration": seg['duration'],
                "Sentiment": seg['sentiment_score'],
                "Topic": f"Segment {seg['id']}"
            })
        
        df_timeline = pd.DataFrame(df_timeline)
        
        # Main timeline
        fig = px.timeline(
            df_timeline,
            x_start="Start",
            x_end="End",
            y="Segment",
            color="Sentiment",
            color_continuous_scale=["red", "yellow", "green"],
            title="🎬 Episode Timeline with Sentiment Coloring",
            height=500
        )
        fig.update_layout(
            xaxis_title="Time (seconds)", 
            yaxis_title="Segments",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment boundaries and markers
        if show_boundaries:
            st.subheader("📍 Segment Boundaries & Markers")
            
            fig2 = go.Figure()
            
            # Add segment boundaries
            for seg in index.segments:
                fig2.add_vline(x=seg['start'], line_dash="dash", line_color="gray", opacity=0.7,
                              annotation_text=f"S{seg['id']}", annotation_position="top")
            
            # Add sentiment markers
            colors = []
            for sent in df_timeline['Sentiment']:
                if sent > 0.5:
                    colors.append('green')
                elif sent < -0.5:
                    colors.append('red')
                else:
                    colors.append('orange')
            
            fig2.add_trace(go.Scatter(
                x=df_timeline['Start'],
                y=[f"S{seg['id']}" for seg in index.segments],
                mode='markers',
                marker=dict(size=12, color=colors, symbol='diamond'),
                name='Segment Start',
                text=[f"Sentiment: {sent:.2f}" for sent in df_timeline['Sentiment']],
                hovertemplate="Segment: %{y}<br>Time: %{x}s<br>%{text}"
            ))
            
            fig2.update_layout(
                title="🎯 Segment Boundaries with Sentiment Markers",
                xaxis_title="Time (seconds)",
                yaxis_title="Segments",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Timeline statistics
        st.subheader("⏱️ Timeline Statistics")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Total Duration", f"{df_timeline['End'].max():.1f}s")
        with stat_col2:
            st.metric("Avg Segment Length", f"{df_timeline['Duration'].mean():.1f}s")
        with stat_col3:
            st.metric("Longest Segment", f"{df_timeline['Duration'].max():.1f}s")
        with stat_col4:
            st.metric("Shortest Segment", f"{df_timeline['Duration'].min():.1f}s")
    
    elif page == "🏷️ Keywords":
        st.header("🏷️ Keyword Analysis & Insights")
        
        # All keywords for analysis
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
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🌈 Word Cloud")
            if filtered_freq:
                # Create word cloud
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    colormap='plasma',
                    max_words=50
                ).generate_from_frequencies(filtered_freq)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("No keywords match the current filters.")
        
        with col2:
            st.subheader("📊 Top Keywords")
            if filtered_freq:
                top_kw = sorted(filtered_freq.items(), key=lambda x: x[1], reverse=True)[:15]
                df_kw = pd.DataFrame(top_kw, columns=['Keyword', 'Frequency'])
                
                fig = px.bar(df_kw, x='Keyword', y='Frequency', 
                           title='Most Frequent Keywords',
                           color='Frequency',
                           color_continuous_scale='Viridis')
                fig.update_layout(xaxis_title='Keyword', yaxis_title='Frequency')
                st.plotly_chart(fig, use_container_width=True)
        
        # Keywords per segment with enhanced table
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
        
        # Keyword network/connections
        if len(filtered_freq) > 1:
            st.subheader("🔗 Keyword Co-occurrence")
            # Simple co-occurrence matrix
            kw_list = list(filtered_freq.keys())[:10]  # Top 10
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
                    st.write(f"• {pair[0]} ↔ {pair[1]}: {count} segments")
    
    elif page == "😊 Sentiment":
        st.header("😊 Sentiment Analysis & Trends")
        
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
        sent_col1, sent_col2 = st.columns([1, 3])
        with sent_col1:
            show_trend = st.checkbox("Show Trend Line", value=True)
        with sent_col2:
            sentiment_view = st.radio("View Type", ["Line Chart", "Bar Chart", "Area Chart"], horizontal=True)
        
        # Main sentiment chart
        if sentiment_view == "Line Chart":
            fig = px.line(
                df_sentiment,
                x='Segment',
                y='Sentiment',
                markers=True,
                title='📈 Sentiment Progression Throughout Episode',
                color_discrete_sequence=["#FF6B6B"],
                height=400
            )
        elif sentiment_view == "Bar Chart":
            fig = px.bar(
                df_sentiment,
                x='Segment',
                y='Sentiment',
                title='📊 Sentiment Scores by Segment',
                color='Sentiment',
                color_continuous_scale=['red', 'yellow', 'green'],
                height=400
            )
        else:  # Area Chart
            fig = px.area(
                df_sentiment,
                x='Segment',
                y='Sentiment',
                title='🌊 Sentiment Flow Throughout Episode',
                color_discrete_sequence=["#4ECDC4"],
                height=400
            )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Neutral")
        if show_trend:
            # Add trend line if sufficient data
            try:
                if len(df_sentiment) > 1:
                    fig.add_trace(px.scatter(df_sentiment, x='Segment', y='Sentiment', trendline="ols").data[1])
            except Exception as trend_err:
                # trendline failed (e.g. insufficient data), ignore
                pass
        fig.update_layout(yaxis_title='Sentiment Score', xaxis_title='Segment')
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment distribution and insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥧 Sentiment Distribution")
            # Sentiment statistics
            pos_count = len(df_sentiment[df_sentiment['Sentiment'] > 0.5])
            neg_count = len(df_sentiment[df_sentiment['Sentiment'] < -0.5])
            neu_count = len(df_sentiment) - pos_count - neg_count
            
            sentiment_dist = {
                '😊 Positive': pos_count,
                '😞 Negative': neg_count,
                '😐 Neutral': neu_count
            }
            
            fig = px.pie(
                values=list(sentiment_dist.values()),
                names=list(sentiment_dist.keys()),
                title='Sentiment Distribution',
                color_discrete_map={'😊 Positive': 'green', '😞 Negative': 'red', '😐 Neutral': 'gray'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Sentiment Insights")
            # Additional metrics
            st.metric("Overall Sentiment", f"{sentiment_score:.3f}")
            st.metric("Sentiment Range", f"{df_sentiment['Sentiment'].min():.2f} to {df_sentiment['Sentiment'].max():.2f}")
            st.metric("Most Positive", f"S{df_sentiment.loc[df_sentiment['Sentiment'].idxmax()]['Segment'][1:]} ({df_sentiment['Sentiment'].max():.2f})")
            st.metric("Most Negative", f"S{df_sentiment.loc[df_sentiment['Sentiment'].idxmin()]['Segment'][1:]} ({df_sentiment['Sentiment'].min():.2f})")
            
            # Sentiment volatility
            volatility = df_sentiment['Sentiment'].std()
            st.metric("Sentiment Volatility", f"{volatility:.3f}")
        
        # Sentiment correlations
        st.subheader(" Sentiment Correlations")
        corr_df = df_sentiment[['Sentiment', 'Duration', 'Word Count']].corr()
        
        fig = px.imshow(corr_df, 
                       text_auto=True, 
                       title="Correlation Matrix: Sentiment vs Other Metrics",
                       color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown
        st.subheader(" Detailed Sentiment Breakdown")
        st.dataframe(df_sentiment.style.apply(lambda x: ['background-color: lightgreen' if v > 0.5 
                                                        else 'background-color: lightcoral' if v < -0.5 
                                                        else 'background-color: lightyellow' for v in x], 
                                             subset=['Sentiment']), 
                   use_container_width=True)
