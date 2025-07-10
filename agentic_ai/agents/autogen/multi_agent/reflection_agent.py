import logging  
from typing import Any  
  
from dotenv import load_dotenv  
  
from autogen_agentchat.agents import AssistantAgent  
from autogen_agentchat.teams import RoundRobinGroupChat  
from autogen_agentchat.conditions import TextMessageTermination  
from autogen_core import CancellationToken  
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient  
from autogen_ext.tools.mcp import SseServerParams, mcp_server_tools  
  
from agents.base_agent import BaseAgent    
  
class Agent(BaseAgent):  
    """  
    Reflection agent utilizing a primary/critic composition in a round-robin chat.  
    """  
  
    def __init__(self, state_store: dict, session_id: str) -> None:  
        super().__init__(state_store, session_id)  
        self.team_agent: Any = None  
        self._initialized: bool = False  
  
    async def _setup_team_agent(self) -> None:  
        if self._initialized:  
            return  
  
        try:  
            server_params = SseServerParams(  
                url=self.mcp_server_uri,  
                headers={"Content-Type": "application/json"},  
                timeout=30,  
            )  
            tools = await mcp_server_tools(server_params)  

            server_params_atlassian = SseServerParams(  
                url="http://localhost:8000/sse",  
                headers={"Content-Type": "application/json"},  
                timeout=30,  
            )  
            tools_atlassian = await mcp_server_tools(server_params) 
            model_client = AzureOpenAIChatCompletionClient(  
                api_key=self.azure_openai_key,  
                azure_endpoint=self.azure_openai_endpoint,  
                api_version=self.api_version,  
                azure_deployment=self.azure_deployment,  
                model=self.openai_model_name,  
            )  
  
            Azure_agent = AssistantAgent(  
                name="AzureSearch",  
                model_client=model_client,  
                tools=tools,  
                system_message=(  
                    "You are a helpful assistant that adds relevant context Azure AI Search to answer questions. You can use multiple tools to find information and answer questions."  
                ),  
            )  
  
            Jira_agent = AssistantAgent(  
                name="JiraAgent",  
                model_client=model_client,  
                tools=tools_atlassian,  
                system_message="You are a helpful assistant that answers questions related to Jira. You can use any tool to find information and answer questions. Do not ask clarifying questions",  
            )  

            Confulence_agent = AssistantAgent(  
                name="ConfluenceAgent",  
                model_client=model_client,  
                tools=tools_atlassian,  
                system_message="You are a helpful assistant that answers questions related to Confluence. You can use any tool to find information and answer questions. Do not ask clarifying questions",  
            ) 

            Final_agent = AssistantAgent(  
                name="Master",  
                model_client=model_client,  
                # tools=[tools_atlassian, tools],  
                system_message="You are a helpful assistant that lists all answers ConfluenceAgent, JiraAgent and AzureSearch and present to user.") 
            termination_condition = TextMessageTermination("Master")  
            self.team_agent = RoundRobinGroupChat(  
                [Azure_agent, Jira_agent,Confulence_agent,Final_agent],  
                termination_condition=termination_condition,  
            )  
  
            if self.state:  
                await self.team_agent.load_state(self.state)  
  
            self._initialized = True  
        except Exception as e:  
            logging.error(f"Error initializing ReflectionAgent: {e}")  
            raise  
  
    async def chat_async(self, prompt: str) -> str:  
        """  
        Run primary/critic group chat and return the final assistant response.  
        """  
        await self._setup_team_agent()  
  
        try:  
            response = await self.team_agent.run(  
                task=prompt,  
                cancellation_token=CancellationToken(),  
            )  
            assistant_response = response.messages[-1].content  
  
            messages = [  
                {"role": "user", "content": prompt},  
                {"role": "assistant", "content": assistant_response},  
            ]  
            self.append_to_chat_history(messages)  
  
            # Save agent's state  
            new_state = await self.team_agent.save_state()  
            self._setstate(new_state)  
  
            return assistant_response  
        except Exception as e:  
            logging.error(f"Error in chat_async: {e}")  
            return "Sorry, an error occurred while processing your request."  
