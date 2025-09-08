from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.utilities import SerpAPIWrapper
from langchain.schema import HumanMessage, SystemMessage
from typing import List, Dict, Optional, Any
import json
import streamlit as st
from config import Config
from utils import ContentExtractor, format_citation
from memory_manager import get_memory_manager, create_research_context_prompt
from dotenv import load_dotenv
import re
from datetime import datetime

load_dotenv()

class ResearchAgent:
    """Main Research Agent that orchestrates the entire workflow with memory"""
    
    def __init__(self, user_id: str = "default_user"):
        self.config = Config()
        self.user_id = user_id
        self.memory = get_memory_manager()
        self.current_session_id = None
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMPERATURE,
            max_output_tokens=self.config.MAX_TOKENS
        )
        self.search = SerpAPIWrapper(serpapi_api_key=self.config.SERPAPI_API_KEY)
        self.extractor = ContentExtractor(timeout=self.config.REQUEST_TIMEOUT)
    
    def research_topic(self, query: str, progress_callback=None, use_memory: bool = True) -> Dict:
        """Main research workflow with memory integration"""
        try:
            # Create new session
            self.current_session_id = self.memory.create_session(self.user_id, query)
            
            # Add user query to conversation history
            self.memory.add_conversation_turn(
                self.current_session_id, 
                "user", 
                f"Research request: {query}"
            )
            
            # Step 1: Check for similar research if memory is enabled
            if use_memory:
                if progress_callback:
                    progress_callback(5, "Checking previous research...")
                
                similar_research = self.memory.find_similar_research(query, self.user_id)
                if similar_research:
                    # Add context from similar research
                    context_info = f"Found {len(similar_research)} similar research topics from your history."
                    self.memory.add_conversation_turn(
                        self.current_session_id,
                        "assistant",
                        context_info,
                        {"similar_research_count": len(similar_research)}
                    )
            
            # Step 2: Web Search
            if progress_callback:
                progress_callback(10, "Searching the web...")
            
            # Create context-aware search if we have previous research
            search_query = self._enhance_query_with_context(query) if use_memory else query
            search_results = self._web_search(search_query)
            
            if not search_results:
                error_msg = "No search results found"
                self.memory.add_conversation_turn(
                    self.current_session_id,
                    "assistant",
                    f"Error: {error_msg}"
                )
                return {"error": error_msg, "status": "error", "session_id": self.current_session_id}
            
            # Step 3: Extract Content
            if progress_callback:
                progress_callback(30, "Extracting content from articles...")
            
            articles = self._extract_articles(search_results, progress_callback)
            
            if not articles:
                error_msg = "No content could be extracted"
                self.memory.add_conversation_turn(
                    self.current_session_id,
                    "assistant",
                    f"Error: {error_msg}"
                )
                return {"error": error_msg, "status": "error", "session_id": self.current_session_id}
            
            # Step 4: Summarize and Analyze with Memory Context
            if progress_callback:
                progress_callback(80, "Analyzing and summarizing content...")
            
            summary = self._generate_summary_with_memory(query, articles)
            
            # Step 5: Save results to memory
            research_results = {
                "query": query,
                "summary": summary,
                "articles": articles,
                "total_sources": len(articles),
                "status": "success",
                "session_id": self.current_session_id,
                "enhanced_query": search_query if search_query != query else None
            }
            
            # Update memory with results
            self.memory.update_research_results(self.current_session_id, research_results)
            
            # Add completion message to conversation
            completion_msg = f"Research completed successfully. Analyzed {len(articles)} sources and generated comprehensive summary."
            self.memory.add_conversation_turn(
                self.current_session_id,
                "assistant",
                completion_msg,
                {
                    "articles_processed": len(articles),
                    "word_count": summary.get("word_count", 0)
                }
            )
            
            if progress_callback:
                progress_callback(100, "Research complete!")
            
            return research_results
            
        except Exception as e:
            error_msg = f"Research failed: {str(e)}"
            if self.current_session_id:
                self.memory.add_conversation_turn(
                    self.current_session_id,
                    "assistant",
                    f"Error: {error_msg}"
                )
            
            return {
                "error": error_msg,
                "status": "error",
                "session_id": self.current_session_id
            }
    
    def _enhance_query_with_context(self, query: str) -> str:
        """Enhance query with context from memory"""
        if not self.current_session_id:
            return query
        
        context = self.memory.get_research_context(self.current_session_id, include_history=False)
        
        # Check user's research history for related topics
        user_history = self.memory.get_user_research_history(self.user_id, limit=5)
        
        if user_history:
            # Extract keywords from previous research
            all_keywords = []
            for session in user_history:
                words = [w for w in session["query"].split() if len(w) > 4]
                all_keywords.extend(words)
            
            # Find relevant keywords for current query
            query_words = set(query.lower().split())
            relevant_keywords = [kw for kw in set(all_keywords) 
                               if any(qw in kw.lower() or kw.lower() in qw 
                                     for qw in query_words)]
            
            if relevant_keywords:
                # Enhance query with related terms
                enhanced_query = f"{query} {' '.join(relevant_keywords[:3])}"
                return enhanced_query
        
        return query
    
    def _web_search(self, query: str) -> List[Dict]:
        """Perform web search and return results"""
        try:
            search_data = self.search.results(query)
            search_results = []
            
            if 'organic_results' in search_data:
                for result in search_data['organic_results'][:self.config.MAX_SEARCH_RESULTS]:
                    # Clean and validate title
                    title = result.get('title', '').strip()
                    if not title:
                        title = self._extract_title_from_url(result.get('link', ''))
                    
                    search_results.append({
                        'title': title or 'Research Article',
                        'url': result.get('link', ''),
                        'snippet': result.get('snippet', ''),
                        'source': result.get('source', 'Web Source')
                    })
            
            return search_results
            
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            return []
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract a readable title from URL"""
        if not url:
            return "Research Article"
        
        try:
            # Remove protocol and www
            clean_url = re.sub(r'https?://(www\.)?', '', url)
            # Get domain
            domain = clean_url.split('/')[0]
            # Get path
            path = clean_url.split('/', 1)[-1] if '/' in clean_url else ''
            
            if path:
                # Extract meaningful part from path
                path_parts = path.replace('-', ' ').replace('_', ' ').split('/')
                for part in path_parts:
                    if len(part) > 3 and not part.isdigit():
                        return part.title()[:50]
            
            return f"Article from {domain.replace('.', ' ').title()}"
            
        except:
            return "Research Article"
    
    def _extract_articles(self, search_results: List[Dict], progress_callback=None) -> List[Dict]:
        """Extract content from search result URLs"""
        articles = []
        total_urls = min(len(search_results), self.config.MAX_ARTICLES_TO_PROCESS)
        
        for i, result in enumerate(search_results[:self.config.MAX_ARTICLES_TO_PROCESS]):
            if progress_callback:
                progress = 30 + (40 * (i + 1) / total_urls)
                progress_callback(progress, f"Extracting article {i+1} of {total_urls}...")
            
            url = result.get('url')
            if not url:
                continue
                
            # Extract content
            article_data = self.extractor.extract_content(
                url, 
                max_length=self.config.MAX_CONTENT_LENGTH
            )
            
            # Fix title issues
            if not article_data.get('title') or article_data.get('title') == 'Unknown Title':
                article_data['title'] = result.get('title', 'Research Article')
            
            # Add search result metadata
            article_data.update({
                'search_title': result.get('title', ''),
                'search_snippet': result.get('snippet', ''),
                'search_source': result.get('source', ''),
                'domain': self._extract_domain(url)
            })
            
            # Only add successful extractions with meaningful content
            if (article_data['status'] == 'success' and 
                len(article_data.get('content', '').strip()) > 100):
                articles.append(article_data)
            
            # For failed extractions, try to use search snippet
            elif len(result.get('snippet', '')) > 50:
                fallback_article = {
                    'title': result.get('title', 'Research Article'),
                    'url': url,
                    'content': result.get('snippet', ''),
                    'domain': self._extract_domain(url),
                    'status': 'partial',
                    'date': 'Recent',
                    'author': 'Web Source'
                }
                articles.append(fallback_article)

        return articles
    
    def _extract_domain(self, url: str) -> str:
        """Extract clean domain name from URL"""
        try:
            domain = re.sub(r'https?://(www\.)?', '', url).split('/')[0]
            return domain
        except:
            return "Unknown Source"
    
    def _generate_summary_with_memory(self, query: str, articles: List[Dict]) -> Dict:
        """Generate comprehensive summary using LLM with memory context"""
        
        # Get memory context
        memory_context = ""
        if self.current_session_id:
            context = self.memory.get_research_context(self.current_session_id)
            
            if context.get("previous_insights"):
                memory_context = "\n\nContext from your previous research:\n"
                for insight in context["previous_insights"][:3]:
                    memory_context += f"• {insight}\n"
            
            # Get user research patterns
            user_insights = self.memory.get_research_insights(self.user_id)
            if user_insights.get("top_topics"):
                memory_context += f"\nYour research interests include: {', '.join([topic[0] for topic in user_insights['top_topics'][:5]])}\n"
        
        # Prepare content for LLM
        articles_text = self._prepare_articles_for_llm(articles)
        
        # Enhanced system prompt with memory awareness
        system_prompt = f"""You are an expert research analyst with access to the user's research history. Create a comprehensive research summary based on the provided articles.

        {memory_context}

        Structure your response as follows:
        
        ## Executive Summary
        A 2-3 sentence overview of the key findings, considering the user's research context.
        
        ## Key Findings
        4-6 main points with evidence from sources. Use [1], [2], etc. to reference sources.
        
        ## Analysis & Insights
        Your analysis of the information, trends, and implications. Connect to previous research interests when relevant.
        
        ## Connections to Previous Research
        If applicable, note how this research relates to the user's previous interests.
        
        ## Conclusion
        A synthesis of the research with key takeaways and suggestions for further research.
        
        Be clear, objective, and cite sources appropriately."""

        user_prompt = f"""Research Topic: "{query}"

        Sources to analyze:
        {articles_text}

        Please provide a comprehensive research summary that builds on the user's research context."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            summary_text = response.content
            
            # Create source mapping
            sources = self._create_source_mapping(articles)
            
            return {
                "summary_text": summary_text,
                "sources": sources,
                "word_count": len(summary_text.split()),
                "articles_analyzed": len(articles),
                "memory_enhanced": bool(memory_context)
            }
            
        except Exception as e:
            return {
                "summary_text": f"Summary generation failed: {str(e)}",
                "sources": self._create_source_mapping(articles),
                "word_count": 0,
                "articles_analyzed": len(articles),
                "memory_enhanced": False
            }
    
    def _prepare_articles_for_llm(self, articles: List[Dict]) -> str:
        """Format articles for LLM input"""
        formatted_articles = []
        
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'Research Article')
            content = article.get('content', '')[:1500]  # Limit content length
            domain = article.get('domain', 'Unknown')
            
            article_text = f"""
[Source {i}]
Title: {title}
Domain: {domain}
URL: {article.get('url', '')}

Content:
{content}

---
"""
            formatted_articles.append(article_text)
        
        return "\n".join(formatted_articles)
    
    def _create_source_mapping(self, articles: List[Dict]) -> List[Dict]:
        """Create formatted source list with citations"""
        sources = []
        
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'Research Article')
            domain = article.get('domain', 'Unknown Source')
            
            citation = format_citation(article, self.config.CITATION_STYLE)
            
            sources.append({
                "number": i,
                "title": title,
                "url": article.get('url', ''),
                "domain": domain,
                "date": article.get('date', 'Recent'),
                "author": article.get('author', domain),
                "citation": citation,
                "status": article.get('status', 'success')
            })
        
        return sources
    
    def generate_full_paper(self, query: str, articles: List[Dict], progress_callback=None) -> Dict:
        """Generate a structured research paper with memory context"""
        try:
            if progress_callback:
                progress_callback(15, "Preparing paper structure...")
            
            # Get memory context for paper generation
            memory_context = ""
            if self.current_session_id:
                context = self.memory.get_research_context(self.current_session_id)
                user_insights = self.memory.get_research_insights(self.user_id)
                
                if context.get("previous_insights") or user_insights.get("top_topics"):
                    memory_context = "\n\nResearch Context:\n"
                    if context.get("previous_insights"):
                        memory_context += "Previous research insights:\n"
                        for insight in context["previous_insights"][:3]:
                            memory_context += f"• {insight}\n"
                    
                    if user_insights.get("top_topics"):
                        memory_context += f"Your research focus areas: {', '.join([topic[0] for topic in user_insights['top_topics'][:5]])}\n"
            
            # Simplified paper structure for hackathon demo
            sections = [
                ("Abstract", "Write a comprehensive abstract (200-300 words) that considers the user's research context"),
                ("Introduction", "Write a detailed introduction with background, objectives, and connection to the user's research interests"),
                ("Literature Review", "Analyze and synthesize the key findings from sources, building on previous research context"),
                ("Discussion", "Discuss implications, applications, and significance in relation to the user's research patterns"),
                ("Conclusion", "Summarize findings and suggest future research directions aligned with user interests")
            ]
            
            full_paper = f"# Research Paper: {query}\n\n"
            full_paper += f"**Generated:** {datetime.now().strftime('%B %d, %Y')}\n"
            full_paper += f"**Sources:** {len(articles)} articles analyzed\n"
            if memory_context:
                full_paper += f"**Research Context:** Enhanced with user's research history\n"
            full_paper += "\n"
            
            # Add table of contents
            full_paper += "## Table of Contents\n"
            for section_name, _ in sections:
                full_paper += f"- [{section_name}](#{section_name.lower().replace(' ', '-')})\n"
            full_paper += "- [References](#references)\n\n"
            
            total_sections = len(sections)
            
            for i, (section_name, section_prompt) in enumerate(sections):
                progress = 20 + (60 * (i + 1) / total_sections)
                if progress_callback:
                    progress_callback(progress, f"Writing {section_name}...")
                
                full_prompt = f"""
                You are writing the {section_name} section of a research paper on: "{query}"
                
                {section_prompt}
                
                {memory_context}
                
                Use the following sources:
                {self._prepare_articles_for_llm(articles[:5])}  # Limit for token management
                
                Requirements:
                - Academic tone and structure
                - 800-1200 words
                - Use markdown formatting
                - Cite sources as [1], [2], etc.
                - Be comprehensive and insightful
                - Consider the user's research context when provided
                """
                
                try:
                    response = self.llm.invoke([HumanMessage(content=full_prompt)])
                    content = response.content if hasattr(response, "content") else str(response)
                    full_paper += f"## {section_name}\n\n{content}\n\n"
                except Exception as e:
                    full_paper += f"## {section_name}\n\n*Error generating this section: {str(e)}*\n\n"
            
            if progress_callback:
                progress_callback(90, "Adding references...")
            
            # Add references
            full_paper += "## References\n\n"
            for i, article in enumerate(articles, 1):
                citation = format_citation(article, self.config.CITATION_STYLE)
                full_paper += f"[{i}] {citation}\n\n"
            
            # Save paper generation to memory
            if self.current_session_id:
                self.memory.add_conversation_turn(
                    self.current_session_id,
                    "assistant",
                    "Generated comprehensive research paper with academic structure and citations.",
                    {
                        "paper_sections": len(sections),
                        "paper_length": len(full_paper),
                        "memory_enhanced": bool(memory_context)
                    }
                )
            
            if progress_callback:
                progress_callback(100, "Paper completed!")
            
            return {
                "query": query,
                "research_paper": full_paper.strip(),
                "articles_used": len(articles),
                "status": "success",
                "session_id": self.current_session_id,
                "memory_enhanced": bool(memory_context)
            }
            
        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Error: {str(e)}")
            
            if self.current_session_id:
                self.memory.add_conversation_turn(
                    self.current_session_id,
                    "assistant",
                    f"Error generating research paper: {str(e)}"
                )
            
            return {
                "research_paper": f"Error generating paper: {str(e)}",
                "status": "error",
                "session_id": self.current_session_id
            }
    
    def get_research_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's research history"""
        return self.memory.get_user_research_history(self.user_id, limit)
    
    def get_research_insights(self) -> Dict[str, Any]:
        """Get insights about user's research patterns"""
        return self.memory.get_research_insights(self.user_id)
    
    def find_related_research(self, query: str) -> List[Dict[str, Any]]:
        """Find related research from user's history"""
        return self.memory.find_similar_research(query, self.user_id)
    
    def continue_research_session(self, session_id: str, additional_query: str) -> Dict:
        """Continue an existing research session"""
        # Load the session
        session = self.memory._load_session(session_id)
        if not session:
            return {"error": "Session not found", "status": "error"}
        
        self.current_session_id = session_id
        
        # Add continuation query
        self.memory.add_conversation_turn(
            session_id,
            "user",
            f"Follow-up research: {additional_query}"
        )
        
        # Perform research with enhanced context
        context_prompt = create_research_context_prompt(session_id, additional_query)
        return self.research_topic(additional_query, use_memory=True)

def create_research_agent(user_id: str = None) -> Optional[ResearchAgent]:
    """Factory function to create research agent with error handling"""
    try:
        # Validate configuration
        missing_keys = Config.validate_config()
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        # Use session state for user ID if available
        if not user_id and hasattr(st, 'session_state') and 'user_id' in st.session_state:
            user_id = st.session_state.user_id
        elif not user_id:
            # Generate a simple user ID
            import hashlib
            user_id = hashlib.md5(f"user_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
            if hasattr(st, 'session_state'):
                st.session_state.user_id = user_id
        
        return ResearchAgent(user_id=user_id)
    
    except Exception as e:
        st.error(f"Failed to initialize Research Agent: {str(e)}")
        return None