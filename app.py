import streamlit as st
import os
from datetime import datetime
import json
from research_chain import create_research_agent
from config import Config

# Page configuration with forced light theme
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional light theme
st.markdown("""
<style>
/* Force light theme */
.stApp {
    background-color: #ffffff;
    color: #262730;
}

.main-header {
    text-align: center;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
    color: white;
    padding: 2.5rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(79, 70, 229, 0.3);
}

.main-header h1 {
    margin: 0;
    font-weight: 700;
    font-size: 2.5rem;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    opacity: 0.95;
    font-size: 1.1rem;
}

.research-card {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    padding: 2rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 5px solid #4f46e5;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
}

.research-card h3 {
    color: #1e293b;
    margin-top: 0;
    font-weight: 600;
}

.source-item {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.source-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}

.source-item h4 {
    color: #1e293b;
    margin-top: 0;
    font-weight: 600;
}

.metric-container {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #bae6fd;
    box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #f8fafc;
}

/* Button styling */
.stButton > button {
    border-radius: 8px;
    border: none;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
}

/* Progress bar styling */
.stProgress > div > div > div {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
}

/* Text area and input styling */
.stTextArea > div > div > textarea {
    border-radius: 8px;
    border: 2px solid #e2e8f0;
    background-color: #ffffff;
}

.stTextInput > div > div > input {
    border-radius: 8px;
    border: 2px solid #e2e8f0;
    background-color: #ffffff;
}

/* Metrics styling */
.metric-container .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0369a1;
}

/* Section dividers */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
    margin: 1.5rem 0;
    border-radius: 2px;
}

/* Status indicators */
.status-success {
    color: #16a34a;
    font-weight: 600;
}

.status-error {
    color: #dc2626;
    font-weight: 600;
}

/* Download section */
.download-section {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #f59e0b;
    margin: 1rem 0;
}

/* Fix spacing issues */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Remove extra gaps */
.element-container {
    margin-bottom: 0.5rem !important;
}

div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div {
    gap: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Enhanced Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI Research Agent</h1>
        <p>Professional research assistant for comprehensive information gathering, analysis, and synthesis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Key inputs with better styling
        if not Config.GOOGLE_API_KEY:
            gemini_key = st.text_input("🔑 Google Gemini API Key", type="password", help="Required for AI summarization")
            if gemini_key:
                os.environ["GOOGLE_API_KEY"] = gemini_key
        
        if not Config.SERPAPI_API_KEY:
            serpapi_key = st.text_input("🔑 SerpAPI Key", type="password", help="Required for web search")
            if serpapi_key:
                os.environ["SERPAPI_API_KEY"] = serpapi_key
        
        st.divider()
        
        # Research settings with better organization
        st.markdown("### 🎛️ Research Settings")
        max_articles = st.slider("📊 Max Articles to Process", 3, 10, Config.MAX_ARTICLES_TO_PROCESS)
        Config.MAX_ARTICLES_TO_PROCESS = max_articles
        
        content_length = st.slider("📝 Content Length per Article", 1000, 5000, Config.MAX_CONTENT_LENGTH)
        Config.MAX_CONTENT_LENGTH = content_length
        
        st.divider()
        
        # Enhanced sample queries
        st.markdown("### 💡 Sample Research Topics")
        sample_queries = [
            "Recent advancements in quantum computing 2024",
            "Benefits and risks of intermittent fasting",
            "How do transformer models work in AI",
            "Latest breakthroughs in renewable energy",
            "Impact of remote work on productivity"
        ]
        
        for i, query in enumerate(sample_queries):
            if st.button(f"📋 {query[:35]}...", key=f"sample_{i}", use_container_width=True):
                st.session_state.research_query = query
    
    # Main content area with better layout
    col1, col2 = st.columns([2.5, 1], gap="large")
    
    with col1:
        st.markdown("### 🔍 Research Query")
        
        default_query = st.session_state.get('research_query', '')
        
        research_query = st.text_area(
            "Enter your research topic or question:",
            value=default_query,
            height=120,
            placeholder="e.g., 'What are the latest developments in artificial intelligence safety and alignment research?'",
            help="Be specific and detailed for better research results"
        )
        
        col_search, col_clear = st.columns([3, 1])
        
        with col_search:
            search_button = st.button("🚀 Start Research", type="primary", use_container_width=True)
        
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.clear()
                st.rerun()
    
    with col2:
        st.markdown("### 📊 Research Statistics")
        
        if 'research_results' in st.session_state:
            results = st.session_state.research_results
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("📚 Sources", results.get('total_sources', 0))
                st.metric("📄 Articles", results.get('summary', {}).get('articles_analyzed', 0))
            with col_m2:
                st.metric("📝 Words", results.get('summary', {}).get('word_count', 0))
                if results.get('research_paper'):
                    st.metric("📑 Paper", "✅")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.info("🎯 Run a research query to see detailed statistics and results")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Research execution with enhanced error handling
    if search_button and research_query.strip():
        missing_keys = Config.validate_config()
        if missing_keys:
            st.error(f"❌ Missing API keys: {', '.join(missing_keys)}")
            st.info("💡 Please add your API keys in the sidebar to continue.")
            return
        
        with st.spinner("🔄 Initializing AI Research Agent..."):
            agent = create_research_agent()
        
        if not agent:
            st.error("❌ Failed to initialize research agent. Please check your API keys.")
            return
        
        # Enhanced progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        def update_progress(percentage, message):
            if percentage > 1:
                percentage = percentage / 100.0
            percentage = max(0.0, min(1.0, percentage))
            progress_bar.progress(percentage)
            status_text.text(f"🔄 {message}")
        
        try:
            results = agent.research_topic(research_query, progress_callback=update_progress)
            st.session_state.research_results = results
            
            progress_bar.empty()
            status_text.empty()
            
            if results.get('status') == 'error':
                st.error(f"❌ Research failed: {results.get('error', 'Unknown error')}")
                return
            
            st.success("✅ Research completed successfully!")
            display_research_results(results)
            
        except Exception as e:
            st.error(f"❌ Research error: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    elif search_button and not research_query.strip():
        st.warning("⚠️ Please enter a research query to proceed.")
    
    elif 'research_results' in st.session_state:
        display_research_results(st.session_state.research_results)

def display_research_results(results):
    """Enhanced display of research results with better structure"""
    
    st.markdown("---")
    st.markdown("## 📋 Research Results")
    
    # Query information card
    st.markdown(f"""
    <div class="research-card">
        <h3>🎯 Research Query</h3>
        <p style="font-size: 1.1rem; font-weight: 500;">"{results.get('query', 'Unknown')}"</p>
        <p style="color: #64748b; margin-bottom: 0;">📅 Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary section with enhanced styling
    summary = results.get('summary', {})
    if summary.get('summary_text'):
        st.markdown("""
        <div class="research-card">
            <h3>📄 Executive Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(summary['summary_text'])

    # Research paper generation section - MOVED UP
    if not results.get('research_paper'):
        st.markdown("""
        <div class="research-card">
            <h3>📑 Generate Complete Research Paper</h3>
            <p>Create a comprehensive academic-style research paper from your sources.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📑 Generate Research Paper", type="primary", use_container_width=True):
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

            def update_progress(percentage, message):
                if percentage > 1:
                    percentage = percentage / 100.0
                percentage = max(0.0, min(1.0, percentage))
                progress_bar.progress(percentage)
                status_text.text(f"🔄 {message}")

            try:
                update_progress(5, "Initializing research paper generation...")
                
                agent = create_research_agent()
                research_query = results.get('query', 'Unknown Query')
                articles = results.get('articles', [])

                update_progress(10, "Starting paper generation...")
                
                # Generate the research paper with progress callback
                research_paper_result = agent.generate_full_paper(
                    research_query, 
                    articles, 
                    progress_callback=update_progress
                )
                
                # Store the full result dictionary, not just the text
                st.session_state.research_results['research_paper'] = research_paper_result

                progress_bar.empty()
                status_text.empty()
                
                st.success("📄 Research paper generated successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Failed to generate research paper: {str(e)}")
                progress_bar.empty()
                status_text.empty()

    # Research paper download section - show if paper exists
    paper_result = results.get("research_paper")
    if paper_result and isinstance(paper_result, dict) and paper_result.get("research_paper"):
        st.markdown("""
        <div class="research-card">
            <h3>📑 Complete Research Paper</h3>
            <p>Your comprehensive research paper is ready for download in multiple formats.</p>
        </div>
        """, unsafe_allow_html=True)

        # Extract the actual paper text from the result dictionary
        paper_text = paper_result.get("research_paper", "")
        
        if paper_text and isinstance(paper_text, str):
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📄 Download Paper (Markdown)",
                    data=paper_text,
                    file_name=f"research_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "📝 Download Paper (Text)",
                    data=paper_text,
                    file_name=f"research_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    # Download section for summary reports
    if summary.get('summary_text'):
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.markdown("#### 💾 Download Research Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            markdown_content = generate_markdown_report(results)
            st.download_button(
                "📄 Markdown Report",
                data=markdown_content,
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            json_content = json.dumps(results, indent=2, default=str)
            st.download_button(
                "📊 JSON Data",
                data=json_content,
                file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            text_content = generate_text_report(results)
            st.download_button(
                "📝 Text Report",
                data=text_content,
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced analysis metrics - MOVED UP
    articles = results.get('articles', [])
    sources = summary.get('sources', [])
    
    if articles:
        st.markdown("## 📊 Research Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Total Sources", len(sources))
        with col2:
            successful_extractions = len([a for a in articles if a.get('status') == 'success'])
            st.metric("✅ Successful Extractions", successful_extractions)
        with col3:
            total_words = sum(len(a.get('content', '').split()) for a in articles)
            st.metric("📝 Words Analyzed", f"{total_words:,}")
        with col4:
            avg_content_length = total_words // len(articles) if articles else 0
            st.metric("📊 Avg. Article Length", avg_content_length)

    # Sources section with better organization - MOVED TO END
    if sources:
        st.markdown(f"## 📚 Research Sources ({len(sources)})")
        
        for source in sources:
            status_icon = "✅"
            status_class = "status-success" if source.get('status') == 'success' else "status-error"
            
            st.markdown(f"""
            <div class="source-item">
                <h4>{status_icon} [{source.get('number', '?')}] {source.get('title', 'Unknown Title')}</h4>
                <p><strong>🌐 Domain:</strong> {source.get('domain', 'Unknown')}</p>
                <p><strong>📅 Date:</strong> {source.get('date', 'Unknown')}</p>
                <p><strong>📖 Citation:</strong> {source.get('citation', 'N/A')}</p>
                <p><a href="{source.get('url', '#')}" target="_blank" style="color: #4f46e5; text-decoration: none; font-weight: 500;">🔗 View Original Article →</a></p>
            </div>
            """, unsafe_allow_html=True)

        # Detailed article analysis in expandable section
        with st.expander("🔍 Detailed Article Analysis"):
            for i, article in enumerate(articles, 1):
                status_emoji = "✅"
                st.markdown(f"**Article {i}:** {status_emoji} **{article.get('status', 'unknown').upper()}**")
                st.markdown(f"- **📄 Title:** {article.get('title', 'Unknown')}")
                st.markdown(f"- **🔗 URL:** {article.get('url', 'N/A')}")
                st.markdown(f"- **🌐 Domain:** {article.get('domain', 'Unknown')}")
                st.markdown(f"- **📊 Content Length:** {len(article.get('content', '')):,} characters")
                if article.get('status') == 'error':
                    st.markdown(f"- **⚠️ Error:** {article.get('error', 'Unknown error')}")
                if i < len(articles):  # Don't add separator after last item
                    st.markdown("---")

def generate_markdown_report(results):
    """Generate enhanced markdown report"""
    query = results.get('query', 'Unknown Query')
    summary = results.get('summary', {})
    sources = summary.get('sources', [])
    
    markdown_content = f"""# 🔍 AI Research Report: {query}

**📅 Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  
**📚 Total Sources:** {len(sources)}  
**📝 Summary Length:** {summary.get('word_count', 0)} words  
**📊 Articles Analyzed:** {summary.get('articles_analyzed', 0)}

---

## 📄 Executive Summary

{summary.get('summary_text', 'No summary available')}

---

## 📚 Research Sources

"""
    
    for source in sources:
        status_icon = "✅"
        markdown_content += f"""### {status_icon} [{source.get('number', '?')}] {source.get('title', 'Unknown Title')}

- **🔗 URL:** {source.get('url', 'N/A')}
- **🌐 Domain:** {source.get('domain', 'Unknown')}
- **📅 Date:** {source.get('date', 'Unknown')}
- **📖 Citation:** {source.get('citation', 'N/A')}

"""
    
    markdown_content += f"""---

## 📊 Research Metadata

- **🔬 Research Method:** AI-powered web search and content analysis
- **🤖 Generated by:** AI Research Agent v2.0
- **⏰ Processing Time:** Automated real-time analysis
- **🎯 Query Complexity:** Advanced multi-source synthesis

---

*This report was generated using advanced AI research techniques for comprehensive information gathering and analysis.*
"""
    
    return markdown_content

def generate_text_report(results):
    """Generate enhanced plain text report"""
    query = results.get('query', 'Unknown Query')
    summary = results.get('summary', {})
    sources = summary.get('sources', [])
    
    text_content = f"""AI RESEARCH REPORT
==================

Research Query: {query}
Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Total Sources: {len(sources)}
Summary Length: {summary.get('word_count', 0)} words
Articles Analyzed: {summary.get('articles_analyzed', 0)}

EXECUTIVE SUMMARY
=================

{summary.get('summary_text', 'No summary available')}

RESEARCH SOURCES
================

"""
    
    for source in sources:
        status = "SUCCESS" if source.get('status') == 'success' else "ERROR"
        text_content += f"""[{source.get('number', '?')}] {source.get('title', 'Unknown Title')} - {status}
URL: {source.get('url', 'N/A')}
Domain: {source.get('domain', 'Unknown')}
Date: {source.get('date', 'Unknown')}
Citation: {source.get('citation', 'N/A')}

"""
    
    text_content += f"""RESEARCH METADATA
=================

Articles Processed: {summary.get('articles_analyzed', 0)}
Research Method: AI-powered web search and content analysis
Generated by: AI Research Agent v2.0
Processing: Automated real-time analysis

This report was generated using advanced AI research techniques.
"""
    
    return text_content

if __name__ == "__main__":
    main()