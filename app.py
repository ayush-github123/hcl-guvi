import streamlit as st
import os
from datetime import datetime
import json
from research_chain import create_research_agent
from memory_manager import get_memory_manager
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

.memory-card {
    background: linear-gradient(135deg, #fef7cd 0%, #fef3c7 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 5px solid #f59e0b;
    border: 1px solid #f3d365;
}

.history-item {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border: 1px solid #e5e7eb;
    cursor: pointer;
    transition: all 0.2s ease;
}

.history-item:hover {
    background-color: #f9fafb;
    border-color: #4f46e5;
    transform: translateX(5px);
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

.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
    color: white;
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

def initialize_user_session():
    """Initialize user session and memory"""
    if 'user_id' not in st.session_state:
        import hashlib
        st.session_state.user_id = hashlib.md5(f"user_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    
    if 'memory_enabled' not in st.session_state:
        st.session_state.memory_enabled = True

def main():
    # Initialize user session
    initialize_user_session()
    
    # Enhanced Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI Research Agent</h1>
        <p>Professional research assistant with memory - remembers your research history and provides contextual insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get memory manager
    memory = get_memory_manager()
    
    # Sidebar for configuration and memory
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
        
        # Memory settings
        st.markdown("### 🧠 Memory Settings")
        memory_enabled = st.checkbox("Enable Research Memory", value=st.session_state.memory_enabled, help="Remember research history and provide context")
        st.session_state.memory_enabled = memory_enabled
        
        if memory_enabled:
            # User insights
            insights = memory.get_research_insights(st.session_state.user_id)
            if insights.get("total_sessions", 0) > 0:
                st.markdown("#### 📊 Your Research Profile")
                st.metric("Total Sessions", insights["total_sessions"])
                st.metric("Recent Activity", insights["recent_activity"])
                
                if insights.get("top_topics"):
                    st.markdown("**🎯 Top Research Areas:**")
                    for topic, count in insights["top_topics"][:3]:
                        st.write(f"• {topic} ({count}x)")
        
        st.divider()
        
        # Research settings
        st.markdown("### 🎛️ Research Settings")
        max_articles = st.slider("📊 Max Articles to Process", 3, 10, Config.MAX_ARTICLES_TO_PROCESS)
        Config.MAX_ARTICLES_TO_PROCESS = max_articles
        
        content_length = st.slider("📝 Content Length per Article", 1000, 5000, Config.MAX_CONTENT_LENGTH)
        Config.MAX_CONTENT_LENGTH = content_length
        
        st.divider()
        
        # Research History
        if memory_enabled:
            st.markdown("### 📚 Research History")
            history = memory.get_user_research_history(st.session_state.user_id, limit=5)
            
            if history:
                st.markdown("**Recent Research:**")
                for idx, session in enumerate(history):
                    with st.container():
                        # Use both index and session_id to ensure uniqueness
                        unique_key = f"hist_{idx}_{session['session_id'][:8]}"
                        if st.button(f"📄 {session['query'][:30]}...", key=unique_key, use_container_width=True):
                            st.session_state.research_query = session['query']
                            st.rerun()
            else:
                st.info("No research history yet. Start your first research!")
        
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
            help="Be specific and detailed for better research results. Memory will help provide context from your previous research."
        )
        
        col_search, col_clear, col_insights = st.columns([2, 1, 1])
        
        with col_search:
            search_button = st.button("🚀 Start Research", type="primary", use_container_width=True)
        
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        with col_insights:
            if memory_enabled and st.button("🔍 Find Similar", use_container_width=True):
                if research_query.strip():
                    similar = memory.find_similar_research(research_query.strip(), st.session_state.user_id)
                    if similar:
                        st.session_state.show_similar = similar
                    else:
                        st.session_state.show_similar = []
        
        # Show similar research if found
        if st.session_state.get('show_similar'):
            st.markdown("#### 🔗 Similar Research Found:")
            for sim in st.session_state.show_similar[:3]:
                st.markdown(f"""
                <div class="history-item">
                    <strong>📄 {sim['query']}</strong><br>
                    <small>Similarity: {sim['similarity']:.0%} | {sim['timestamp'][:10]}</small>
                </div>
                """, unsafe_allow_html=True)
    
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
                if results.get('summary', {}).get('memory_enhanced'):
                    st.metric("🧠 Memory", "✅")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show session info
            if results.get('session_id'):
                st.markdown(f"**🔗 Session:** `{results['session_id'][:8]}...`")
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
            agent = create_research_agent(st.session_state.user_id)
        
        if not agent:
            st.error("❌ Failed to initialize research agent. Please check your API keys.")
            return
        
        # Check for similar research
        if memory_enabled:
            similar_research = agent.find_related_research(research_query)
            if similar_research:
                st.info(f"🔍 Found {len(similar_research)} similar research topics in your history. This will help provide better context!")
        
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
            results = agent.research_topic(research_query, progress_callback=update_progress, use_memory=memory_enabled)
            st.session_state.research_results = results
            
            progress_bar.empty()
            status_text.empty()
            
            if results.get('status') == 'error':
                st.error(f"❌ Research failed: {results.get('error', 'Unknown error')}")
                return
            
            success_msg = "✅ Research completed successfully!"
            if results.get('summary', {}).get('memory_enhanced'):
                success_msg += " 🧠 Enhanced with your research history!"
            
            st.success(success_msg)
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
    """Enhanced display of research results with memory features"""
    
    st.markdown("---")
    st.markdown("## 📋 Research Results")
    
    # Query information card with memory indicators
    memory_indicator = ""
    if results.get('summary', {}).get('memory_enhanced'):
        memory_indicator = " 🧠 *Enhanced with research memory*"
    
    session_info = ""
    if results.get('session_id'):
        session_info = f"**🔗 Session ID:** `{results['session_id']}`\n"
    
    st.markdown(f"""
    <div class="research-card">
        <h3>🎯 Research Query{memory_indicator}</h3>
        <p style="font-size: 1.1rem; font-weight: 500;">"{results.get('query', 'Unknown')}"</p>
        <p style="color: #64748b; margin-bottom: 0;">📅 Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        {session_info}
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced query info
    if results.get('enhanced_query') and results['enhanced_query'] != results.get('query'):
        st.markdown(f"""
        <div class="memory-card">
            <h4>🧠 Memory-Enhanced Search</h4>
            <p>Your search was enhanced with context from previous research:</p>
            <p><em>"{results['enhanced_query']}"</em></p>
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

    # Research paper generation section
    if not results.get('research_paper'):
        st.markdown("""
        <div class="research-card">
            <h3>📑 Generate Complete Research Paper</h3>
            <p>Create a comprehensive academic-style research paper from your sources.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_paper, col_continue = st.columns(2)
        
        with col_paper:
            if st.button("📑 Generate Research Paper", type="primary", use_container_width=True):
                generate_research_paper(results)
        
        with col_continue:
            if results.get('session_id') and st.button("➕ Continue Research", type="secondary", use_container_width=True):
                st.session_state.continue_session_id = results['session_id']
                st.session_state.show_continue_form = True

    # Continue research form
    if st.session_state.get('show_continue_form'):
        st.markdown("""
        <div class="memory-card">
            <h3>➕ Continue Research Session</h3>
        </div>
        """, unsafe_allow_html=True)
        
        continue_query = st.text_area(
            "Additional research question:",
            placeholder="e.g., 'What are the practical applications of this research?'",
            key="continue_query"
        )
        
        col_cont, col_cancel = st.columns(2)
        with col_cont:
            if st.button("🚀 Continue Research", use_container_width=True):
                if continue_query.strip():
                    continue_research_session(continue_query.strip())
        with col_cancel:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_continue_form = False
                st.rerun()

    # Research paper download section
    paper_result = results.get("research_paper")
    if paper_result and isinstance(paper_result, dict) and paper_result.get("research_paper"):
        st.markdown("""
        <div class="research-card">
            <h3>📑 Complete Research Paper</h3>
            <p>Your comprehensive research paper is ready for download in multiple formats.</p>
        </div>
        """, unsafe_allow_html=True)

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

    # Enhanced analysis metrics
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

    # Memory insights section
    if st.session_state.memory_enabled and results.get('session_id'):
        memory = get_memory_manager()
        user_insights = memory.get_research_insights(st.session_state.user_id)
        
        if user_insights.get('insights'):
            st.markdown("## 🧠 Research Insights")
            
            st.markdown("""
            <div class="memory-card">
                <h4>📈 Your Research Patterns</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for insight in user_insights['insights'][:3]:
                st.markdown(f"• {insight}")

    # Sources section with better organization
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
                if i < len(articles):
                    st.markdown("---")

def generate_research_paper(results):
    """Generate research paper with progress tracking"""
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
        
        agent = create_research_agent(st.session_state.user_id)
        research_query = results.get('query', 'Unknown Query')
        articles = results.get('articles', [])

        update_progress(10, "Starting paper generation...")
        
        research_paper_result = agent.generate_full_paper(
            research_query, 
            articles, 
            progress_callback=update_progress
        )
        
        st.session_state.research_results['research_paper'] = research_paper_result

        progress_bar.empty()
        status_text.empty()
        
        success_msg = "📄 Research paper generated successfully!"
        if research_paper_result.get('memory_enhanced'):
            success_msg += " 🧠 Enhanced with your research context!"
        
        st.success(success_msg)
        st.rerun()

    except Exception as e:
        st.error(f"❌ Failed to generate research paper: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def continue_research_session(continue_query):
    """Continue an existing research session"""
    session_id = st.session_state.get('continue_session_id')
    if not session_id:
        st.error("❌ No active session found")
        return
    
    try:
        agent = create_research_agent(st.session_state.user_id)
        
        with st.spinner("🔄 Continuing research session..."):
            continued_results = agent.continue_research_session(session_id, continue_query)
            
            if continued_results.get('status') == 'success':
                st.session_state.research_results = continued_results
                st.session_state.show_continue_form = False
                st.success("✅ Research session continued successfully! 🧠 Building on previous context.")
                st.rerun()
            else:
                st.error(f"❌ Failed to continue research: {continued_results.get('error', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"❌ Error continuing research: {str(e)}")

def generate_markdown_report(results):
    """Generate enhanced markdown report with memory indicators"""
    query = results.get('query', 'Unknown Query')
    summary = results.get('summary', {})
    sources = summary.get('sources', [])
    
    memory_note = ""
    if summary.get('memory_enhanced'):
        memory_note = " (Enhanced with Research Memory 🧠)"
    
    session_note = ""
    if results.get('session_id'):
        session_note = f"**🔗 Session ID:** `{results['session_id']}`  \n"
    
    markdown_content = f"""# 🔍 AI Research Report: {query}{memory_note}

**📅 Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  
{session_note}**📚 Total Sources:** {len(sources)}  
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
- **🤖 Generated by:** AI Research Agent v2.0 with Memory
- **⏰ Processing Time:** Automated real-time analysis
- **🎯 Query Complexity:** Advanced multi-source synthesis
- **🧠 Memory Enhancement:** {"Yes" if summary.get('memory_enhanced') else "No"}

---

*This report was generated using advanced AI research techniques with contextual memory for enhanced insights.*
"""
    
    return markdown_content

def generate_text_report(results):
    """Generate enhanced plain text report with memory indicators"""
    query = results.get('query', 'Unknown Query')
    summary = results.get('summary', {})
    sources = summary.get('sources', [])
    
    memory_indicator = " (Memory Enhanced)" if summary.get('memory_enhanced') else ""
    
    text_content = f"""AI RESEARCH REPORT{memory_indicator}
==================

Research Query: {query}
Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Session ID: {results.get('session_id', 'N/A')}
Total Sources: {len(sources)}
Summary Length: {summary.get('word_count', 0)} words
Articles Analyzed: {summary.get('articles_analyzed', 0)}
Memory Enhanced: {"Yes" if summary.get('memory_enhanced') else "No"}

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
Generated by: AI Research Agent v2.0 with Memory
Processing: Automated real-time analysis
Memory Enhancement: {"Enabled" if summary.get('memory_enhanced') else "Disabled"}

This report was generated using advanced AI research techniques with contextual memory.
"""
    
    return text_content

if __name__ == "__main__":
    main()