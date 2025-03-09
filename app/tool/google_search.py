import asyncio
from typing import List
from baidusearch.baidusearch import search  # 更改导入的模块
from app.tool.base import BaseTool


class GoogleSearch(BaseTool):  # 保持类名不变
    name: str = "baidu_search"  # 修改工具名称为百度搜索
    description: str = """Perform a Baidu search and return a list of relevant links.
Use this tool when you need to find information on the web, get up-to-date data, or research specific topics.
The tool returns a list of URLs that match the search query.
"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query to submit to Baidu.",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, num_results: int = 10) -> List[str]:
        """
        Execute a Baidu search and return a list of URLs.

        Args:
            query (str): The search query to submit to Baidu.
            num_results (int, optional): The number of search results to return. Default is 10.

        Returns:
            List[str]: A list of URLs matching the search query.
        """
        # Run the search in a thread pool to prevent blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: search(query, num_results=num_results)
        )

        # Extract URLs from the search results
        links = [result['url'] for result in results]

        return links