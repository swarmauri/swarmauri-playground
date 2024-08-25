#!/usr/bin/env python3
"""Hello!
To add a new LLM, you need to:
    1. Import the LLM class
    2. Define the API key for the LLM
    3. Add the LLM class to the `available_llms` dictionary
    4. Add the LLM class to the `llms` dictionary

Then you are done! The new LLM will be available in the dropdown menu.

P.S: Follow the structure of the existing LLMs to add a new one.
"""

import os
from dotenv import load_dotenv
import gradio as gr
from swarmauri.standard.llms.concrete.GroqModel import GroqModel
from swarmauri.standard.llms.concrete.OpenAIModel import OpenAIModel
from swarmauri.standard.agents.concrete.SimpleConversationAgent import (
    SimpleConversationAgent,
)
from swarmauri.standard.conversations.concrete.Conversation import Conversation

# Load the environment variables
load_dotenv()

# Fetch the API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define the available LLMs
available_llms = {
    "GroqModel": (GroqModel, GROQ_API_KEY),
    "OpenAIModel": (OpenAIModel, OPENAI_API_KEY),
}

# Define the allowed models for each LLM
llms = {
    "GroqModel": [
        model
        for model in GroqModel(api_key=available_llms["GroqModel"][1]).allowed_models
    ],
    "OpenAIModel": [
        model
        for model in OpenAIModel(
            api_key=available_llms["OpenAIModel"][1]
        ).allowed_models
    ],
}

# Initialize Conversation
conversation = Conversation()


# Define the default values and global variables
CHOSEN_LLM = available_llms["GroqModel"][0]
DEFAULT_AI = "GroqModel"
DEFAULT_MODEL = llms["GroqModel"][0]
API_KEY = available_llms["GroqModel"][1]
TEMPERATURE = 0.7
MAX_TOKENS = 512


# Define the callback function for the LLM component dropdown
def llm_component_callback(component):
    """Shows all the models available for the selected LLM component"""
    if llms.get(component, None) is not None:
        global CHOSEN_LLM
        global API_KEY
        CHOSEN_LLM = available_llms[component][0]
        API_KEY = available_llms[component][1]
        return gr.update(choices=[model for model in llms[component]])
    return gr.update(choices=[])


# Define the callback function for the LLM model dropdown
def llm_model_callback(model):
    return f"Agent created with {model}"


# Function to handle conversation
def handle_conversation(llm_model, user_message, history, temperature, max_tokens):
    """Main function which handles the conversation"""
    agent = SimpleConversationAgent(
        llm=CHOSEN_LLM(
            name=llm_model,
            api_key=API_KEY,
        ),
        conversation=conversation,
    )
    llm_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = agent.exec(
        user_message,
        llm_kwargs=llm_kwargs,
    )
    history.append((user_message, response))
    return history, "", history


# Define the function to update the code preview
import ast


def update_code_preview():
    """
    In this function, we read the current file and collect the relevant lines
    until the specified comment is reached. We then parse the response with AST
    and filter out the nodes based on the `to_be_ignored` set. Finally, we convert
    the cleaned AST back to source code and return the cleaned code.
    """
    response_lines = []

    # Read the current file and collect relevant lines
    with open(__file__, "r", encoding="utf-8") as f:
        documentation_start = False
        first = True

        for line in f:
            stripped_line = line.strip()

            # Check if the line starts or ends the documentation string
            if first and '"""' in stripped_line:
                documentation_start = not documentation_start
                first = False
                continue

            if documentation_start and '"""' in stripped_line:
                documentation_start = not documentation_start
                continue

            if documentation_start:
                continue

            # Stop collecting lines when reaching the specified comment
            if stripped_line == "# Define the function to update the code preview":
                break

            # Collect the line
            response_lines.append(line)

    response = "".join(response_lines)

    # Parse the response with AST
    tree = ast.parse(response)

    # Define the names of nodes to be ignored
    to_be_ignored = {
        "llm_model_callback",
        "available_llms",
        "llms",
        "CHOSEN_LLM",
    }

    # Filter out nodes based on the `to_be_ignored` set
    class CodeCleaner(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if any(pattern in node.name for pattern in to_be_ignored):
                return None
            return node

        def visit_Assign(self, node):
            if any(
                isinstance(target, ast.Name)
                and any(pattern in target.id for pattern in to_be_ignored)
                for target in node.targets
            ):
                return None
            return node

    # Transform the AST to remove ignored nodes
    cleaned_tree = CodeCleaner().visit(tree)

    # Convert the cleaned AST back to source code
    cleaned_code = ast.unparse(cleaned_tree)

    # Return the cleaned code
    return cleaned_code


# Create the interface within a Blocks context
with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column(scale=1):
            llm_component_dropdown = gr.Dropdown(
                choices=[model for model in available_llms.keys()],
                label="LLM Component",
                value=DEFAULT_AI,
            )
            llm_model_dropdown = gr.Dropdown(
                choices=llms[DEFAULT_AI],
                label="LLM Model",
                value=DEFAULT_MODEL,
            )

            temperature_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=TEMPERATURE,
                step=0.01,
                label="Temperature",
            )

            max_tokens_slider = gr.Slider(
                minimum=1, maximum=2048, value=MAX_TOKENS, step=1, label="Max Tokens"
            )

            # Set up the event to update the model dropdown when LLM component changes
            llm_component_dropdown.change(
                fn=llm_component_callback,
                inputs=llm_component_dropdown,
                outputs=llm_model_dropdown,
            )

            output_text = gr.Textbox(label="Output")

            llm_model_dropdown.change(
                fn=llm_model_callback, inputs=llm_model_dropdown, outputs=output_text
            )

        with gr.Column(scale=4):
            chat_interface = gr.Chatbot(label="Conversation History", container=False)

            user_input = gr.Textbox(label="Your message")

            # Update chat history based on user input
            user_input.submit(
                fn=handle_conversation,
                inputs=[
                    llm_model_dropdown,
                    user_input,
                    chat_interface,
                    temperature_slider,
                    max_tokens_slider,
                ],
                outputs=[chat_interface, user_input, chat_interface],
                scroll_to_output=True,  # This will scroll the chat interface to the bottom
            )

        # Add a column for the code preview block
        with gr.Column(scale=3):
            code_preview = gr.Code(
                language="python",
                label="Code Preview",
                value=update_code_preview(),
            )

# Run the interface
if __name__ == "__main__":
    interface.title = "Swarmauri Playground"
    interface.launch()
