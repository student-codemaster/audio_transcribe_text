"""Visualization components for podcast analysis dashboard."""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import numpy as np


def create_segment_timeline(segments):
    """Create interactive timeline visualization for segments.
    
    Args:
        segments: List of segment dictionaries with 'id', 'start', 'end', 'summary', 'keywords'
    
    Returns:
        Plotly figure object
    """
    if not segments:
        return go.Figure().add_annotation(text="No segments available")
    
    # Prepare data
    segment_data = []
    for seg in segments:
        segment_data.append({
            'id': seg.get('id', ''),
            'start': seg.get('start', 0),
            'end': seg.get('end', 0),
            'duration': seg.get('duration', seg.get('end', 0) - seg.get('start', 0)),
            'summary': seg.get('summary', ''),
            'keywords': ', '.join(seg.get('keywords', [])) if seg.get('keywords') else 'N/A'
        })
    
    df = pd.DataFrame(segment_data)
    
    # Create Gantt-like chart
    fig = go.Figure()
    
    for idx, row in df.iterrows():
        fig.add_trace(go.Bar(
            y=[row['id']],
            x=[row['duration']],
            orientation='h',
            name=f"Segment {row['id']}",
            marker=dict(
                color=f"hsl({(idx * 360 / len(df)) % 360}, 70%, 50%)",
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            hovertemplate=(
                f"<b>Segment {row['id']}</b><br>" +
                f"Start: {format_time(row['start'])}<br>" +
                f"End: {format_time(row['end'])}<br>" +
                f"Duration: {row['duration']:.1f}s<br>" +
                f"<b>Summary:</b> {row['summary']}<br>" +
                f"<b>Keywords:</b> {row['keywords']}" +
                "<extra></extra>"
            ),
            text=f"Seg {row['id']}",
            textposition='inside'
        ))
    
    fig.update_layout(
        title="🎬 Episode Timeline - Segment Distribution",
        xaxis_title="Duration (seconds)",
        yaxis_title="Segments",
        showlegend=False,
        height=400,
        template="plotly_white",
        hovermode='closest'
    )
    
    return fig


def create_sentiment_trend(segments):
    """Create sentiment trend visualization.
    
    Args:
        segments: List of segment dictionaries with 'start', 'end', 'sentiment_score'
    
    Returns:
        Plotly figure object
    """
    if not segments:
        return go.Figure().add_annotation(text="No sentiment data available")
    
    segment_data = []
    for seg in segments:
        segment_data.append({
            'mid_time': (seg.get('start', 0) + seg.get('end', 0)) / 2,
            'start': seg.get('start', 0),
            'end': seg.get('end', 0),
            'sentiment': seg.get('sentiment_score', 0),
            'id': seg.get('id', ''),
            'summary': seg.get('summary', '')
        })
    
    df = pd.DataFrame(segment_data)
    df = df.sort_values('mid_time')
    
    # Create line chart with sentiment zones
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(go.Scatter(
        x=df['mid_time'],
        y=df['sentiment'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(
            color='rgb(0, 100, 200)',
            width=3
        ),
        marker=dict(
            size=8,
            color=df['sentiment'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Sentiment", thickness=15),
            line=dict(color='white', width=2)
        ),
        hovertemplate=(
            "<b>Segment %{text}</b><br>" +
            "Time: %{x:.1f}s<br>" +
            "Sentiment: %{y:.3f}<br>" +
            "%{customdata}" +
            "<extra></extra>"
        ),
        text=df['id'],
        customdata=df['summary']
    ))
    
    # Add positive/negative zones
    max_time = df['mid_time'].max() if len(df) > 0 else 1
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="green", opacity=0.3, 
                  annotation_text="Positive Zone", annotation_position="right")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="red", opacity=0.3,
                  annotation_text="Negative Zone", annotation_position="right")
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.2)
    
    fig.update_layout(
        title="😊 Sentiment Trend Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Sentiment Score",
        height=400,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig


def create_keyword_cloud(segments, num_keywords=20):
    """Create word cloud from segment keywords.
    
    Args:
        segments: List of segment dictionaries with 'keywords'
        num_keywords: Max keywords to include
    
    Returns:
        Matplotlib figure object
    """
    # Aggregate all keywords
    all_keywords = {}
    for seg in segments:
        keywords = seg.get('keywords', [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',')]
        for kw in keywords:
            all_keywords[kw] = all_keywords.get(kw, 0) + 1
    
    if not all_keywords:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No keywords available', ha='center', va='center')
        return fig
    
    # Sort and limit
    top_keywords = dict(sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:num_keywords])
    
    # Create word cloud
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if top_keywords:
        wc = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(top_keywords)
        
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
    
    fig.suptitle('🏷️ Top Keywords & Topics in Episode', fontsize=16, fontweight='bold')
    return fig


def create_keyword_bar_chart(segments, num_keywords=15):
    """Create bar chart of top keywords.
    
    Args:
        segments: List of segment dictionaries with 'keywords'
        num_keywords: Number of top keywords to show
    
    Returns:
        Plotly figure object
    """
    # Aggregate keywords with frequency
    keyword_freq = {}
    for seg in segments:
        keywords = seg.get('keywords', [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',')]
        for kw in keywords:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
    
    if not keyword_freq:
        return go.Figure().add_annotation(text="No keywords available")
    
    # Sort and limit
    top_kw = dict(sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:num_keywords])
    
    df = pd.DataFrame(list(top_kw.items()), columns=['keyword', 'frequency'])
    
    fig = px.bar(
        df,
        x='frequency',
        y='keyword',
        orientation='h',
        title=f"🏷️ Top {num_keywords} Keywords in Episode",
        labels={'frequency': 'Frequency', 'keyword': 'Keyword'},
        color='frequency',
        color_continuous_scale='Viridis',
        height=400 + (len(df) * 15)
    )
    
    fig.update_layout(
        showlegend=False,
        template="plotly_white",
        hovermode='closest'
    )
    
    return fig


def create_segment_distribution(segments):
    """Create pie chart of segment distribution.
    
    Args:
        segments: List of segment dictionaries with 'id', 'duration', 'keywords'
    
    Returns:
        Plotly figure object
    """
    if not segments:
        return go.Figure().add_annotation(text="No segment data available")
    
    segment_data = []
    for seg in segments:
        segment_data.append({
            'id': f"Segment {seg.get('id', '')}",
            'duration': seg.get('duration', 0),
            'keywords': ', '.join(seg.get('keywords', [])) if seg.get('keywords') else 'None'
        })
    
    df = pd.DataFrame(segment_data)
    
    fig = px.pie(
        df,
        values='duration',
        names='id',
        title="⏱️ Segment Duration Distribution",
        hover_data=['keywords']
    )
    
    fig.update_layout(height=400, template="plotly_white")
    
    return fig


def create_sentiment_heatmap(segments):
    """Create heatmap of sentiment across segments.
    
    Args:
        segments: List of segment dictionaries
    
    Returns:
        Plotly figure object
    """
    if not segments:
        return go.Figure().add_annotation(text="No sentiment data available")
    
    # Prepare data
    segment_ids = [f"Seg {seg.get('id', '')}" for seg in segments]
    sentiments = [seg.get('sentiment_score', 0) for seg in segments]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[sentiments],
        x=segment_ids,
        y=['Sentiment Score'],
        colorscale='RdYlGn',
        colorbar=dict(title="Score", thickness=20),
        hovertemplate="<b>%{x}</b><br>Sentiment: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="😊 Sentiment Score Heatmap Across Segments",
        height=200,
        template="plotly_white"
    )
    
    return fig


def create_keywords_per_segment(segments, top_n_segments=10):
    """Create visualization of top keywords in each segment.
    
    Args:
        segments: List of segment dictionaries
        top_n_segments: Number of segments to show
    
    Returns:
        Plotly figure object
    """
    if not segments:
        return go.Figure().add_annotation(text="No segment data available")
    
    segments_to_show = segments[:top_n_segments]
    
    fig = go.Figure()
    
    for seg in segments_to_show:
        keywords = seg.get('keywords', [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',')]
        
        keyword_str = ', '.join(keywords[:5]) if keywords else 'None'
        
        fig.add_trace(go.Bar(
            y=[f"Segment {seg.get('id', '')}"],
            x=[1],
            name=keyword_str,
            orientation='h',
            hovertemplate=(
                f"<b>Segment {seg.get('id', '')}</b><br>" +
                f"Keywords: {keyword_str}<br>" +
                f"Summary: {seg.get('summary', '')}" +
                "<extra></extra>"
            ),
            marker=dict(
                color=seg.get('sentiment_score', 0),
                colorscale='RdYlGn',
                showscale=(seg == segments_to_show[0]),
                colorbar=dict(title="Sentiment")
            )
        ))
    
    fig.update_layout(
        title=f"🏷️ Top Keywords (First {top_n_segments} Segments)",
        showlegend=False,
        height=300 + (len(segments_to_show) * 30),
        template="plotly_white",
        xaxis=dict(visible=False)
    )
    
    return fig


def plot_segments(segments):
    """Legacy function - Create basic topic distribution plot.
    
    Args:
        segments: List of segment dictionaries
    
    Returns:
        Matplotlib figure object
    """
    topics = [s.get("topic", f"Topic {i}") for i, s in enumerate(segments)]
    
    df = pd.DataFrame({"topic": topics})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df["topic"].value_counts().plot(kind="bar", ax=ax, color='skyblue', edgecolor='black')
    
    ax.set_title("Topic Distribution")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"