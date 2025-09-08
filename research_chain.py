from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.utilities import SerpAPIWrapper
from langchain.schema import HumanMessage, SystemMessage
from typing import List, Dict, Optional
import json
import streamlit as st
from config import Config
from utils import ContentExtractor, format_citation
from dotenv import load_dotenv
import re
from datetime import datetime

load_dotenv()

class ResearchAgent:
    """Main Research Agent that orchestrates the entire workflow"""
    
    def __init__(self):
        self.config = Config()
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMPERATURE,
            max_output_tokens=self.config.MAX_TOKENS
        )
        self.search = SerpAPIWrapper(serpapi_api_key=self.config.SERPAPI_API_KEY)
        self.extractor = ContentExtractor(timeout=self.config.REQUEST_TIMEOUT)
    
    def research_topic(self, query: str, progress_callback=None) -> Dict:
        """Main research workflow"""
        try:
            # Step 1: Web Search
            if progress_callback:
                progress_callback(10, "Searching the web...")
            
            search_results = self._web_search(query)
            
            if not search_results:
                return {"error": "No search results found", "status": "error"}
            
            # Step 2: Extract Content
            if progress_callback:
                progress_callback(30, "Extracting content from articles...")
            
            articles = self._extract_articles(search_results, progress_callback)
            
            if not articles:
                return {"error": "No content could be extracted", "status": "error"}
            
            # Step 3: Summarize and Analyze
            if progress_callback:
                progress_callback(80, "Analyzing and summarizing content...")
            
            summary = self._generate_summary(query, articles)
            
            if progress_callback:
                progress_callback(100, "Research complete!")
            
            return {
                "query": query,
                "summary": summary,
                "articles": articles,
                "total_sources": len(articles),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": f"Research failed: {str(e)}",
                "status": "error"
            }
    
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
    
    def _generate_summary(self, query: str, articles: List[Dict]) -> Dict:
        """Generate comprehensive summary using LLM"""
        
        # Prepare content for LLM
        articles_text = self._prepare_articles_for_llm(articles)
        
        # Simplified prompt for better results
        system_prompt = """You are an expert research analyst. Create a comprehensive research summary based on the provided articles.

        Structure your response as follows:
        
        ## Executive Summary
        A 2-3 sentence overview of the key findings.
        
        ## Key Findings
        4-6 main points with evidence from sources. Use [1], [2], etc. to reference sources.
        
        ## Analysis & Insights
        Your analysis of the information, trends, and implications.
        
        ## Conclusion
        A synthesis of the research with key takeaways.
        
        Be clear, objective, and cite sources appropriately."""

        user_prompt = f"""Research Topic: "{query}"

        Sources to analyze:
        {articles_text}

        Please provide a comprehensive research summary."""

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
                "articles_analyzed": len(articles)
            }
            
        except Exception as e:
            return {
                "summary_text": f"Summary generation failed: {str(e)}",
                "sources": self._create_source_mapping(articles),
                "word_count": 0,
                "articles_analyzed": len(articles)
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
        """Generate a structured research paper"""
        try:
            if progress_callback:
                progress_callback(15, "Preparing paper structure...")
            
            # Simplified paper structure for hackathon demo
            sections = [
                ("Abstract", "Write a comprehensive abstract (200-300 words)"),
                ("Introduction", "Write a detailed introduction with background and objectives"),
                ("Literature Review", "Analyze and synthesize the key findings from sources"),
                ("Discussion", "Discuss implications, applications, and significance"),
                ("Conclusion", "Summarize findings and suggest future directions")
            ]
            
            full_paper = f"# Research Paper: {query}\n\n"
            full_paper += f"**Generated:** {datetime.now().strftime('%B %d, %Y')}\n"
            full_paper += f"**Sources:** {len(articles)} articles analyzed\n\n"
            
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
                
                Use the following sources:
                {self._prepare_articles_for_llm(articles[:5])}  # Limit for token management
                
                Requirements:
                - Academic tone and structure
                - 800-1200 words
                - Use markdown formatting
                - Cite sources as [1], [2], etc.
                - Be comprehensive and insightful
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
            
            if progress_callback:
                progress_callback(100, "Paper completed!")
            
            return {
                "query": query,
                "research_paper": full_paper.strip(),
                "articles_used": len(articles),
                "status": "success"
            }
            
        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Error: {str(e)}")
            return {
                "research_paper": f"Error generating paper: {str(e)}",
                "status": "error"
            }

def create_research_agent() -> Optional[ResearchAgent]:
    """Factory function to create research agent with error handling"""
    try:
        # Validate configuration
        missing_keys = Config.validate_config()
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        return ResearchAgent()
    
    except Exception as e:
        st.error(f"Failed to initialize Research Agent: {str(e)}")
        return None