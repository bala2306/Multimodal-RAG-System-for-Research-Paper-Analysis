"""Database connection and utilities."""

from typing import Optional
from supabase import create_client, Client
from app.core.config import settings
from loguru import logger


class SupabaseClient:
    """Supabase client singleton."""

    _instance: Optional[Client] = None
    
    @classmethod
    def get_client(cls) -> Client:
        """Get or create Supabase client instance."""
        if cls._instance is None:
            try:
                cls._instance = create_client(
                    settings.supabase_url,
                    settings.supabase_key
                )
                logger.info("Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                raise
        return cls._instance


def get_supabase() -> Client:
    """Dependency for getting Supabase client."""
    return SupabaseClient.get_client()
