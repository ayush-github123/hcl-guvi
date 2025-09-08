import requests
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urlparse
import re
from datetime import datetime
from typing import Dict
import streamlit as st

class ContentExtractor:
    """Handles web content extraction and cleaning"""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        # Set user agent to avoid blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def extract_content(self, url: str, max_length: int = 3000) -> Dict:
        """Extract clean content from a URL"""
        try:
            downloaded = trafilatura.fetch_url(url)
            content = trafilatura.extract(downloaded) if downloaded else None

            response = None
            if not content:
                # Fallback to BeautifulSoup
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                content = self._extract_with_bs4(soup)

            # Extract metadata
            metadata = self._extract_metadata(response.text if response else "", url)

            clean_content = self._clean_text(content or "")
            if len(clean_content) > max_length:
                clean_content = clean_content[:max_length] + "..."

            return {
                'url': url,
                'content': clean_content,
                'title': metadata.get('title', 'Unknown Title'),
                'author': metadata.get('author', urlparse(url).netloc),
                'date': metadata.get('date', 'Unknown Date'),
                'domain': urlparse(url).netloc,
                'status': 'success' if clean_content else 'error'
            }

        except Exception as e:
            return {
                'url': url,
                'content': '',
                'title': 'Failed to Extract',
                'author': 'Unknown',
                'date': 'Unknown',
                'domain': urlparse(url).netloc,
                'status': 'error',
                'error': str(e)
            }

    
    def _extract_with_bs4(self, soup: BeautifulSoup) -> str:
        """Fallback content extraction with BeautifulSoup"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()
        
        # Try to find main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|article|post'))
        
        if main_content:
            return main_content.get_text(strip=True, separator=' ')
        else:
            return soup.get_text(strip=True, separator=' ')
    
    def _extract_metadata(self, html: str, url: str) -> Dict:
        """Extract metadata from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        metadata = {}
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        else:
            # Try meta title
            meta_title = soup.find('meta', attrs={'property': 'og:title'}) or soup.find('meta', attrs={'name': 'title'})
            if meta_title:
                metadata['title'] = meta_title.get('content', '').strip()
        
        # Author
        author_meta = soup.find('meta', attrs={'name': 'author'}) or soup.find('meta', attrs={'property': 'article:author'})
        if author_meta:
            metadata['author'] = author_meta.get('content', '').strip()
        
        # Date
        date_meta = (
            soup.find('meta', attrs={'property': 'article:published_time'}) or 
            soup.find('meta', attrs={'name': 'date'}) or 
            soup.find('time')
        )
        if date_meta:
            date_content = date_meta.get('content') or date_meta.get('datetime') or date_meta.get_text()
            metadata['date'] = self._parse_date(date_content)
        
        return metadata
    
    def _parse_date(self, date_str: str) -> str:
        """Parse and format date string"""
        try:
            # Handle ISO format
            if 'T' in date_str:
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return date_obj.strftime('%B %d, %Y')
            return date_str.strip()
        except:
            return date_str.strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Remove common unwanted patterns
        text = re.sub(r'Cookie Policy.*?Accept', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Subscribe.*?Newsletter', '', text, flags=re.IGNORECASE)
        
        return text.strip()

def format_citation(article_data: Dict, style: str = "APA") -> str:
    """Format citation in specified style"""
    title = article_data.get('title', 'Unknown Title')
    author = article_data.get('author', 'Unknown Author')
    date = article_data.get('date', 'Unknown Date')
    url = article_data.get('url', '')
    domain = article_data.get('domain', '')
    
    if style.upper() == "APA":
        return f"{author}. ({date}). {title}. {domain}. {url}"
    elif style.upper() == "MLA":
        return f'{author}. "{title}." {domain}, {date}, {url}.'
    else:
        return f"{title} - {author} ({date}) - {url}"

def display_progress_bar():
    """Display a progress bar for processing"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    return progress_bar, status_text

def safe_get_text(element, default: str = "Unknown") -> str:
    """Safely extract text from HTML element"""
    if element:
        return element.get_text().strip() or default
    return default
