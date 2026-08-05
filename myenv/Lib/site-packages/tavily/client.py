import os
import requests
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
import time
from .exceptions import *


class TavilyClient:
    def __init__(
            self,
            api_key: Optional[str] = None,
            base_url: str = "https://api.tavily.com",
            timeout: int = 60
    ):
        """
        Initialize Tavily API client

        Args:
            api_key: Tavily API key. If not provided, will look for TAVILY_API_KEY environment variable
            base_url: Base URL for Tavily API
            timeout: Request timeout in seconds (max 120)
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError(
                "API key is required. Set TAVILY_API_KEY environment variable or pass api_key parameter")

        self.base_url = base_url
        self.timeout = min(timeout, 120)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # Track API usage statistics
        self._usage_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "last_request_time": None
        }

    def _update_usage_stats(self, success: bool = True):
        """Update internal usage statistics"""
        self._usage_stats["total_requests"] += 1
        if success:
            self._usage_stats["successful_requests"] += 1
        else:
            self._usage_stats["failed_requests"] += 1
        self._usage_stats["last_request_time"] = datetime.now()

    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get current usage statistics for this client instance

        Returns:
            Dictionary containing usage statistics
        """
        return self._usage_stats.copy()

    def search(
            self,
            query: str,
            search_depth: str = "basic",
            max_results: int = 5,
            include_domains: Optional[List[str]] = None,
            exclude_domains: Optional[List[str]] = None,
            include_answer: bool = False,
            include_raw_content: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Perform a web search using Tavily API

        Args:
            query: Search query string
            search_depth: Search depth level ("basic" or "advanced")
            max_results: Maximum number of results to return
            include_domains: List of domains to include in search
            exclude_domains: List of domains to exclude from search
            include_answer: Whether to include AI-generated answer
            include_raw_content: Whether to include raw content
            **kwargs: Additional parameters to pass to API

        Returns:
            Dictionary containing search results
        """
        data = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
        }

        if include_domains:
            data["include_domains"] = include_domains
        if exclude_domains:
            data["exclude_domains"] = exclude_domains

        data.update(kwargs)

        try:
            response = requests.post(
                f"{self.base_url}/search",
                json=data,
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                self._update_usage_stats(success=True)
                return response.json()
            else:
                self._update_usage_stats(success=False)
                self._handle_error(response)

        except requests.exceptions.Timeout:
            self._update_usage_stats(success=False)
            raise TimeoutError(f"Request timed out after {self.timeout} seconds")
        except requests.exceptions.RequestException as e:
            self._update_usage_stats(success=False)
            raise TavilyError(f"Request failed: {str(e)}")

    def extract(
            self,
            urls: List[str],
            include_images: bool = False,
            format: str = "markdown",
            **kwargs
    ) -> Dict[str, Any]:
        """
        Extract content from URLs using Tavily API

        Args:
            urls: List of URLs to extract content from
            include_images: Whether to include images in extraction
            format: Output format ("markdown", "html", or "text")
            **kwargs: Additional parameters to pass to API

        Returns:
            Dictionary containing extracted content
        """
        data = {
            "urls": urls,
            "include_images": include_images,
            "format": format,
        }
        data.update(kwargs)

        try:
            response = requests.post(
                f"{self.base_url}/extract",
                json=data,
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                self._update_usage_stats(success=True)
                return response.json()
            else:
                self._update_usage_stats(success=False)
                self._handle_error(response)

        except requests.exceptions.Timeout:
            self._update_usage_stats(success=False)
            raise TimeoutError(f"Request timed out after {self.timeout} seconds")
        except requests.exceptions.RequestException as e:
            self._update_usage_stats(success=False)
            raise TavilyError(f"Request failed: {str(e)}")

    def qna_search(
            self,
            query: str,
            search_depth: str = "advanced",
            max_results: int = 5,
            **kwargs
    ) -> str:
        """
        Perform a Q&A style search and return only the answer

        Args:
            query: Question to search for
            search_depth: Search depth level
            max_results: Maximum number of results
            **kwargs: Additional parameters

        Returns:
            AI-generated answer string
        """
        result = self.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=True,
            **kwargs
        )
        return result.get("answer", "")

    def batch_search(
            self,
            queries: List[str],
            search_depth: str = "basic",
            max_results: int = 5,
            delay_between_requests: float = 1.0,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform multiple searches in sequence with optional delay

        Args:
            queries: List of search queries
            search_depth: Search depth level for all queries
            max_results: Maximum results per query
            delay_between_requests: Delay in seconds between requests
            **kwargs: Additional parameters for all searches

        Returns:
            List of search results for each query
        """
        results = []
        for query in queries:
            try:
                result = self.search(
                    query=query,
                    search_depth=search_depth,
                    max_results=max_results,
                    **kwargs
                )
                results.append(result)

                # Add delay between requests to be respectful to API
                if delay_between_requests > 0:
                    time.sleep(delay_between_requests)

            except Exception as e:
                # Log failed search but continue with others
                results.append({"error": str(e), "query": query})

        return results

    def validate_api_key(self) -> bool:
        """
        Validate the API key by making a simple search request

        Returns:
            Boolean indicating if API key is valid
        """
        try:
            # Make a simple test search
            test_result = self.search("test", max_results=1)
            return True
        except (InvalidAPIKeyError, ForbiddenError):
            return False
        except Exception:
            # Other exceptions might not be related to API key validity
            return False

    def get_search_suggestions(
            self,
            query: str,
            max_suggestions: int = 5
    ) -> List[str]:
        """
        Generate search query suggestions based on input query

        Args:
            query: Initial query string
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of suggested search queries
        """
        # This is a mock implementation that could be enhanced
        # with actual API endpoint if available
        base_suggestions = [
            f"{query} tutorial",
            f"{query} guide",
            f"{query} examples",
            f"best {query}",
            f"{query} vs alternatives"
        ]

        return base_suggestions[:max_suggestions]

    def _handle_error(self, response: requests.Response):
        """
        Handle API response errors and raise appropriate exceptions

        Args:
            response: Response object from requests library
        """
        try:
            error_detail = response.json().get("detail", {})
            error_msg = error_detail.get("error", response.text)
        except:
            error_msg = response.text

        if response.status_code == 401:
            raise InvalidAPIKeyError(f"Invalid API key: {error_msg}")
        elif response.status_code == 429:
            raise UsageLimitExceededError(f"Usage limit exceeded: {error_msg}")
        elif response.status_code == 400:
            raise BadRequestError(f"Bad request: {error_msg}")
        elif response.status_code in [403, 432, 433]:
            raise ForbiddenError(f"Forbidden: {error_msg}")
        else:
            response.raise_for_status()

    def __repr__(self) -> str:
        """String representation of TavilyClient instance"""
        return f"TavilyClient(base_url='{self.base_url}', requests_made={self._usage_stats['total_requests']})"

    def __str__(self) -> str:
        """User-friendly string representation"""
        stats = self._usage_stats
        return (f"Tavily API Client\n"
                f"Base URL: {self.base_url}\n"
                f"Total Requests: {stats['total_requests']}\n"
                f"Successful: {stats['successful_requests']}\n"
                f"Failed: {stats['failed_requests']}")