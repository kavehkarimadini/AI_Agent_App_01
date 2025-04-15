from langchain_community.tools import WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

# Save to File Tool
def save_to_txt(data: str, filename: str = "research_output.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(data)
    return f"Data saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    #must not have spaces in the name
    name="DuckDuckGoSearch",
    func=search.run,
    description="A tool to search the web using DuckDuckGo.",
)
api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)