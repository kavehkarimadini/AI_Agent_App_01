from langchain_community.tools import WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

search = DuckDuckGoSearchRun()
search_tool = Tool(
    #must not have spaces in the name
    name="DuckDuckGoSearch",
    func=search.run,
    description="A tool to search the web using DuckDuckGo.",
)