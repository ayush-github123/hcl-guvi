import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    
    # Search Configuration
    MAX_SEARCH_RESULTS = 8
    MAX_ARTICLES_TO_PROCESS = 6
    
    # Content Extraction
    MAX_CONTENT_LENGTH = 3000  # Characters per article
    REQUEST_TIMEOUT = 10
    
    # LLM Configuration
    LLM_MODEL = "gemini-2.0-flash"  # or "gpt-4" for better results
    LLM_TEMPERATURE = 0.3
    MAX_TOKENS = 2000
    
    # Output Configuration
    CITATION_STYLE = "APA"  # APA, MLA, etc.
    
    @staticmethod
    def validate_config():
        """Validate that all required API keys are present"""
        missing_keys = []
        
        if not Config.GOOGLE_API_KEY:
            missing_keys.append("GOOGLE_API_KEY")
        if not Config.SERPAPI_API_KEY:
            missing_keys.append("SERPAPI_API_KEY")
            
        return missing_keys