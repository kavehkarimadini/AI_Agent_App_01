from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool
# Load environment variables
load_dotenv()

class ResearchPaper(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat_model = ChatHuggingFace(llm=llm)
parser = PydanticOutputParser(pydantic_object=ResearchPaper)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
tools = [search_tool]
# Create the agent with the tools and prompt
agent = create_tool_calling_agent(
    llm=chat_model,
    tools=tools,
    prompt=prompt
)
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
user_input = input("what do you want to research about: ")
# Query the LLM
raw_response = agent_executor.invoke({"query": user_input})
try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response.topic)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)