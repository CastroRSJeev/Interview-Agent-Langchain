import os
import aiohttp
import json
from typing import Optional, List, Dict, Any
from .exceptions import *


class AsyncTavilyClient:
    def __init__(
            self,
            api_key: Optional[str] = None,
            base_url: str = "https://api.tavily.com",
            timeout: int = 60
    ):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError(
                "API key is required. Set TAVILY_API_KEY environment variable or pass api_key parameter")

        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=min(timeout, 120))
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    async def search(
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
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                        f"{self.base_url}/search",
                        json=data,
                        headers=self.headers
                ) as response:

                    if response.status == 200:
                        return await response.json()
                    else:
                        await self._handle_error(response)

        except aiohttp.ClientTimeout:
            raise TimeoutError(f"Request timed out after {self.timeout.total} seconds")
        except aiohttp.ClientError as e:
            raise TavilyError(f"Request failed: {str(e)}")

    async def extract(
            self,
            urls: List[str],
            include_images: bool = False,
            format: str = "markdown",
            **kwargs
    ) -> Dict[str, Any]:

        data = {
            "urls": urls,
            "include_images": include_images,
            "format": format,
        }
        data.update(kwargs)

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                        f"{self.base_url}/extract",
                        json=data,
                        headers=self.headers
                ) as response:

                    if response.status == 200:
                        return await response.json()
                    else:
                        await self._handle_error(response)

        except aiohttp.ClientTimeout:
            raise TimeoutError(f"Request timed out after {self.timeout.total} seconds")
        except aiohttp.ClientError as e:
            raise TavilyError(f"Request failed: {str(e)}")

    async def qna_search(
            self,
            query: str,
            search_depth: str = "advanced",
            max_results: int = 5,
            **kwargs
    ) -> str:

        result = await self.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=True,
            **kwargs
        )
        return result.get("answer", "")

    async def _handle_error(self, response: aiohttp.ClientResponse):
        try:
            error_data = await response.json()
            error_detail = error_data.get("detail", {})
            error_msg = error_detail.get("error", await response.text())
        except:
            error_msg = await response.text()

        if response.status == 401:
            raise InvalidAPIKeyError(f"Invalid API key: {error_msg}")
        elif response.status == 429:
            raise UsageLimitExceededError(f"Usage limit exceeded: {error_msg}")
        elif response.status == 400:
            raise BadRequestError(f"Bad request: {error_msg}")
        elif response.status in [403, 432, 433]:
            raise ForbiddenError(f"Forbidden: {error_msg}")
        else:
            raise TavilyError(f"HTTP {response.status}: {error_msg}")