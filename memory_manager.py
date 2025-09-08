"""
Memory Manager for AI Research Agent
Uses modern LangChain approach with custom persistence instead of deprecated memory classes
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import pickle

@dataclass
class ResearchSession:
    """Represents a research session with memory"""
    session_id: str
    user_id: str
    query: str
    timestamp: datetime
    research_results: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@dataclass
class ConversationTurn:
    """Represents a single conversation turn"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

class ResearchMemoryManager:
    """
    Modern memory management for research agent using custom persistence
    Replaces deprecated LangChain memory classes
    """
    
    def __init__(self, storage_path: str = "research_memory", max_sessions: int = 100):
        self.storage_path = Path(storage_path)
        self.max_sessions = max_sessions
        self.storage_path.mkdir(exist_ok=True)
        
        # In-memory cache for active sessions
        self._active_sessions: Dict[str, ResearchSession] = {}
        self._load_recent_sessions()
    
    def create_session(self, user_id: str, query: str) -> str:
        """Create a new research session"""
        session_id = self._generate_session_id(user_id, query)
        
        session = ResearchSession(
            session_id=session_id,
            user_id=user_id,
            query=query,
            timestamp=datetime.now(),
            research_results={},
            conversation_history=[],
            metadata={"created_at": datetime.now().isoformat()}
        )
        
        self._active_sessions[session_id] = session
        self._save_session(session)
        
        return session_id
    
    def add_conversation_turn(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """Add a conversation turn to the session"""
        if session_id not in self._active_sessions:
            self._load_session(session_id)
        
        if session_id in self._active_sessions:
            turn = ConversationTurn(
                role=role,
                content=content,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            self._active_sessions[session_id].conversation_history.append(asdict(turn))
            self._save_session(self._active_sessions[session_id])
    
    def update_research_results(self, session_id: str, results: Dict[str, Any]):
        """Update research results for a session"""
        if session_id not in self._active_sessions:
            self._load_session(session_id)
        
        if session_id in self._active_sessions:
            self._active_sessions[session_id].research_results = results
            self._active_sessions[session_id].metadata["last_updated"] = datetime.now().isoformat()
            self._save_session(self._active_sessions[session_id])
    
    def get_session_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get conversation history for a session"""
        if session_id not in self._active_sessions:
            self._load_session(session_id)
        
        if session_id in self._active_sessions:
            return self._active_sessions[session_id].conversation_history
        
        return None
    
    def get_research_context(self, session_id: str, include_history: bool = True) -> Dict[str, Any]:
        """Get research context for LLM prompts"""
        if session_id not in self._active_sessions:
            self._load_session(session_id)
        
        if session_id not in self._active_sessions:
            return {}
        
        session = self._active_sessions[session_id]
        context = {
            "current_query": session.query,
            "previous_results": session.research_results,
            "session_metadata": session.metadata
        }
        
        if include_history and session.conversation_history:
            # Get last 10 conversation turns for context
            recent_history = session.conversation_history[-10:]
            context["conversation_history"] = recent_history
            
            # Extract key insights from previous conversations
            context["previous_insights"] = self._extract_insights_from_history(recent_history)
        
        return context
    
    def get_user_research_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's research history"""
        user_sessions = []
        
        # Check active sessions
        for session in self._active_sessions.values():
            if session.user_id == user_id:
                user_sessions.append({
                    "session_id": session.session_id,
                    "query": session.query,
                    "timestamp": session.timestamp.isoformat(),
                    "has_results": bool(session.research_results)
                })
        
        # Load additional sessions from storage if needed
        if len(user_sessions) < limit:
            stored_sessions = self._load_user_sessions(user_id, limit - len(user_sessions))
            user_sessions.extend(stored_sessions)
        
        # Sort by timestamp (newest first) and limit
        user_sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        return user_sessions[:limit]
    
    def find_similar_research(self, query: str, user_id: str = None, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find similar research queries"""
        similar_sessions = []
        query_words = set(query.lower().split())
        
        # Search in active sessions
        for session in self._active_sessions.values():
            if user_id and session.user_id != user_id:
                continue
                
            session_words = set(session.query.lower().split())
            similarity = len(query_words.intersection(session_words)) / len(query_words.union(session_words))
            
            if similarity >= similarity_threshold:
                similar_sessions.append({
                    "session_id": session.session_id,
                    "query": session.query,
                    "similarity": similarity,
                    "timestamp": session.timestamp.isoformat(),
                    "has_results": bool(session.research_results)
                })
        
        similar_sessions.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_sessions[:5]
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """Remove old sessions to save space"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # Clean active sessions
        sessions_to_remove = []
        for session_id, session in self._active_sessions.items():
            if session.timestamp < cutoff_date:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self._active_sessions[session_id]
        
        # Clean stored sessions
        for file_path in self.storage_path.glob("*.pkl"):
            try:
                with open(file_path, 'rb') as f:
                    session = pickle.load(f)
                    if session.timestamp < cutoff_date:
                        file_path.unlink()
            except:
                # Remove corrupted files
                file_path.unlink()
    
    def export_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export session data for backup or analysis"""
        if session_id not in self._active_sessions:
            self._load_session(session_id)
        
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            return {
                "session": asdict(session),
                "export_timestamp": datetime.now().isoformat()
            }
        
        return None
    
    def get_research_insights(self, user_id: str) -> Dict[str, Any]:
        """Generate insights from user's research patterns"""
        user_sessions = self.get_user_research_history(user_id, limit=50)
        
        if not user_sessions:
            return {"total_sessions": 0, "insights": []}
        
        # Analyze research patterns
        total_sessions = len(user_sessions)
        topics = {}
        
        for session in user_sessions:
            words = session["query"].lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    topics[word] = topics.get(word, 0) + 1
        
        # Top research topics
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_sessions": total_sessions,
            "top_topics": top_topics,
            "recent_activity": len([s for s in user_sessions if 
                                 datetime.fromisoformat(s["timestamp"]) > 
                                 datetime.now() - timedelta(days=7)]),
            "insights": self._generate_user_insights(user_sessions, top_topics)
        }
    
    def _generate_session_id(self, user_id: str, query: str) -> str:
        """Generate unique session ID"""
        content = f"{user_id}_{query}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _save_session(self, session: ResearchSession):
        """Save session to persistent storage"""
        file_path = self.storage_path / f"{session.session_id}.pkl"
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(session, f)
        except Exception as e:
            print(f"Error saving session {session.session_id}: {e}")
    
    def _load_session(self, session_id: str) -> Optional[ResearchSession]:
        """Load session from persistent storage"""
        file_path = self.storage_path / f"{session_id}.pkl"
        
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    session = pickle.load(f)
                    self._active_sessions[session_id] = session
                    return session
            except Exception as e:
                print(f"Error loading session {session_id}: {e}")
        
        return None
    
    def _load_recent_sessions(self, limit: int = 20):
        """Load recent sessions into memory"""
        session_files = list(self.storage_path.glob("*.pkl"))
        session_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for file_path in session_files[:limit]:
            try:
                with open(file_path, 'rb') as f:
                    session = pickle.load(f)
                    self._active_sessions[session.session_id] = session
            except:
                continue
    
    def _load_user_sessions(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Load user sessions from storage"""
        user_sessions = []
        
        for file_path in self.storage_path.glob("*.pkl"):
            try:
                with open(file_path, 'rb') as f:
                    session = pickle.load(f)
                    if session.user_id == user_id:
                        user_sessions.append({
                            "session_id": session.session_id,
                            "query": session.query,
                            "timestamp": session.timestamp.isoformat(),
                            "has_results": bool(session.research_results)
                        })
            except:
                continue
        
        return user_sessions
    
    def _extract_insights_from_history(self, history: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from conversation history"""
        insights = []
        
        for turn in history:
            if turn["role"] == "assistant" and len(turn["content"]) > 100:
                # Extract first sentence as insight
                sentences = turn["content"].split('. ')
                if sentences:
                    insight = sentences[0].strip()
                    if len(insight) > 20 and insight not in insights:
                        insights.append(insight)
        
        return insights[:5]  # Return top 5 insights
    
    def _generate_user_insights(self, sessions: List[Dict], top_topics: List[tuple]) -> List[str]:
        """Generate insights about user's research patterns"""
        insights = []
        
        if not sessions:
            return insights
        
        # Research frequency insight
        if len(sessions) > 10:
            insights.append(f"You've conducted {len(sessions)} research sessions, showing strong research engagement.")
        
        # Topic focus insight
        if top_topics:
            main_topic = top_topics[0][0]
            insights.append(f"Your research frequently focuses on '{main_topic}' related topics.")
        
        # Recent activity insight
        recent_sessions = [s for s in sessions if 
                         datetime.fromisoformat(s["timestamp"]) > 
                         datetime.now() - timedelta(days=7)]
        
        if len(recent_sessions) > 3:
            insights.append("High research activity this week - you're staying current with information.")
        
        return insights

# Singleton instance for global access
research_memory = ResearchMemoryManager()

def get_memory_manager() -> ResearchMemoryManager:
    """Get the global memory manager instance"""
    return research_memory

def create_research_context_prompt(session_id: str, new_query: str) -> str:
    """Create a context-aware prompt using memory"""
    memory = get_memory_manager()
    context = memory.get_research_context(session_id)
    
    if not context:
        return f"Research Query: {new_query}"
    
    prompt_parts = [f"Current Research Query: {new_query}"]
    
    if context.get("previous_results"):
        prompt_parts.append("\nPrevious Research Context:")
        prev_query = context.get("current_query", "")
        if prev_query:
            prompt_parts.append(f"- Previous query: {prev_query}")
        
        if context.get("previous_insights"):
            prompt_parts.append("- Key insights from previous research:")
            for insight in context["previous_insights"][:3]:
                prompt_parts.append(f"  • {insight}")
    
    if context.get("conversation_history"):
        recent_turns = context["conversation_history"][-4:]  # Last 2 exchanges
        if recent_turns:
            prompt_parts.append("\nRecent Conversation Context:")
            for turn in recent_turns:
                role_emoji = "👤" if turn["role"] == "user" else "🤖"
                content_preview = turn["content"][:100] + "..." if len(turn["content"]) > 100 else turn["content"]
                prompt_parts.append(f"{role_emoji} {content_preview}")
    
    prompt_parts.append("\nPlease consider this context when conducting the new research.")
    
    return "\n".join(prompt_parts)