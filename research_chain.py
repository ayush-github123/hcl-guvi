from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.utilities import SerpAPIWrapper
from langchain.schema import HumanMessage, SystemMessage
from typing import List, Dict, Optional
import json
import streamlit as st
from config import Config
from utils import ContentExtractor, format_citation
from dotenv import load_dotenv

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
            # print("++++++++++++++++SEARCH RESULTS+++++++++++++++++++++", search_results)
            
            if not search_results:
                return {"error": "No search results found", "results": []}
            
            # Step 2: Extract Content
            if progress_callback:
                progress_callback(30, "Extracting content from articles...")
            
            articles = self._extract_articles(search_results, progress_callback)
            
            if not articles:
                return {"error": "No content could be extracted", "results": []}
            
            # Step 3: Summarize and Analyze
            if progress_callback:
                progress_callback(80, "Analyzing and summarizing content...")
            
            summary = self._generate_summary(query, articles)
            # paper = self.generate_full_paper(query, articles)

            # print("SUMMARY --- ", summary)
            
            if progress_callback:
                progress_callback(100, "Research complete!")
            
            return {
                "query": query,
                "summary": summary,
                # "research_paper": paper,
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
            # Use SerpAPI to search
            results = self.search.run(query)
            
            # Parse results (SerpAPI returns string, need to extract URLs)
            search_results = []
            
            # Try to get structured results using SerpAPI's results method
            search_data = self.search.results(query)
            
            if 'organic_results' in search_data:
                for result in search_data['organic_results'][:self.config.MAX_SEARCH_RESULTS]:
                    search_results.append({
                        'title': result.get('title', 'No Title'),
                        'url': result.get('link', ''),
                        'snippet': result.get('snippet', ''),
                        'source': result.get('source', 'Unknown')
                    })
            # print(search_results)
            return search_results
            
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            return []
    
    def _extract_articles(self, search_results: List[Dict], progress_callback=None) -> List[Dict]:
        """Extract content from search result URLs"""
        articles = []
        total_urls = min(len(search_results), self.config.MAX_ARTICLES_TO_PROCESS)
        
        for i, result in enumerate(search_results[:self.config.MAX_ARTICLES_TO_PROCESS]):
            if progress_callback:
                progress = 30 + (40 * (i + 1) / total_urls)  # Progress from 30% to 70%
                progress_callback(progress, f"Extracting article {i+1} of {total_urls}...")
            
            url = result.get('url')
            if not url:
                continue
                
            # Extract content
            article_data = self.extractor.extract_content(
                url, 
                max_length=self.config.MAX_CONTENT_LENGTH
            )
            # print("article data : ", article_data)
            
            # Add search result metadata
            article_data.update({
                'search_title': result.get('title', ''),
                'search_snippet': result.get('snippet', ''),
                'search_source': result.get('source', '')
            })
            
            # Only add successful extractions with meaningful content
            if (article_data['status'] == 'success' and 
                len(article_data['content'].strip()) > 100):
                articles.append(article_data)

            print("\n\nDEBUG:", url, article_data.get("status"), len(article_data.get("content", "")))

        # print(articles)
        return articles
    
    def _generate_summary(self, query: str, articles: List[Dict]) -> Dict:
        """Generate comprehensive summary using LLM"""
        
        # Prepare content for LLM
        articles_text = self._prepare_articles_for_llm(articles)
        print("+++++++++++++++++article text+++++++++++++++", articles_text)
        # Create prompt
        system_prompt = """You are an expert research analyst. Your task is to analyze the provided articles and create a comprehensive, well-structured research summary.

        Please provide:
        1. **Executive Summary**: 2-3 sentence overview of key findings
        2. **Key Findings**: 4-6 main points with supporting evidence and sources mentioned
        3. **Different Perspectives**: Any contrasting viewpoints or debates found
        4. **Recent Developments**: Latest trends or breakthroughs mentioned
        5. **Conclusion**: Synthesis of information and implications

        For each key point, include the source number in brackets [1], [2], etc.

        Format your response in clear markdown with proper headers and bullet points.
        Be objective, accurate, and cite sources appropriately."""

        user_prompt = f"""Research Query: "{query}"

        Articles to analyze:

        {articles_text}

        Please provide a comprehensive research summary based on these sources."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            print(response)
            summary_text = response.content
            
            # Process citations and create source mapping
            sources = self._create_source_mapping(articles)
            
            return {
                "summary_text": summary_text,
                "sources": sources,
                "word_count": len(summary_text.split()),
                "articles_analyzed": len(articles)
            }
            
        except Exception as e:
            return {
                "summary_text": f"Failed to generate summary: {str(e)}",
                "sources": [],
                "word_count": 0,
                "articles_analyzed": len(articles)
            }
    
    def _prepare_articles_for_llm(self, articles: List[Dict]) -> str:
        """Format articles for LLM input"""
        formatted_articles = []
        
        for i, article in enumerate(articles, 1):
            article_text = f"""
            [Source {i}]
            Title: {article.get('title', 'Unknown Title')}
            URL: {article.get('url', '')}
            Date: {article.get('date', 'Unknown Date')}
            Domain: {article.get('domain', 'Unknown Domain')}

            Content:
            {article.get('content', '')[:2000]}...

            ---
            """
            formatted_articles.append(article_text)
        
        return "\n".join(formatted_articles)
    
    def _create_source_mapping(self, articles: List[Dict]) -> List[Dict]:
        """Create formatted source list with citations"""
        sources = []
        
        for i, article in enumerate(articles, 1):
            citation = format_citation(article, self.config.CITATION_STYLE)
            
            sources.append({
                "number": i,
                "title": article.get('title', 'Unknown Title'),
                "url": article.get('url', ''),
                "domain": article.get('domain', ''),
                "date": article.get('date', 'Unknown Date'),
                "author": article.get('author', 'Unknown Author'),
                "citation": citation,
                "status": article.get('status', 'unknown')
            })
        
        return sources
    

    def generate_full_paper(self, query: str, articles: List[Dict]) -> Dict:
        """Generate a structured full-length academic research paper"""
        try:
            SECTION_SUBPARTS = {
                "Title": 1,
                "Abstract": 1,
                "Introduction": 2,
                "Literature Review": 3,
                "Methodology": 2,
                "Results": 2,
                "Discussion": 2,
                "Applications": 1,
                "Conclusion": 1
            }

            def generate_table_of_contents():
                toc = ["# Table of Contents"]
                for section, parts in SECTION_SUBPARTS.items():
                    for part_num in range(1, parts + 1):
                        toc.append(f"- [{section} - Part {part_num}](#{section.lower().replace(' ', '-')}-part-{part_num})")
                toc.append("- [References](#references)")
                return "\n".join(toc)

            full_paper = generate_table_of_contents()

            for section, parts in SECTION_SUBPARTS.items():
                for part_num in range(1, parts + 1):
                    prompt = f"""
                    You are an expert academic writer.

                    Write Part {part_num} of the **{section}** section of a research paper on the topic:

                    "{query}"

                    Use only the following context (from extracted articles):

                    {" ".join([a['content'][:1500] for a in articles])}

                    Requirements:
                    - Formal academic tone
                    - Depth, clarity, structure
                    - No repetition from earlier parts
                    - Aim for ~2000 words
                    - Use markdown, with bullet points/tables if helpful
                    """
                    response = self.llm.invoke([HumanMessage(content=prompt)])
                    content = response.content if hasattr(response, "content") else str(response)
                    
                    full_paper += f"\n\n## {section} - Part {part_num}\n\n{content}"

            # References from your citation util
            refs = [format_citation(a, self.config.CITATION_STYLE) for a in articles]
            references_md = "\n\n## References\n" + "\n".join(refs)
            full_paper += "\n\n" + references_md

            return {
                "query": query,
                "research_paper": full_paper.strip(),
                "articles_used": len(articles),
                "status": "success"
            }

        except Exception as e:
            return {
                "research_paper": f"Error generating paper: {str(e)}",
                "status": "error"
            }
        

def create_research_agent() -> ResearchAgent:
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