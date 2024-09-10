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
import gradio as gr
from swarmauri.standard.llms.concrete.GroqModel import GroqModel
from swarmauri.standard.llms.concrete.OpenAIModel import OpenAIModel
from swarmauri.standard.agents.concrete.SimpleConversationAgent import (
    SimpleConversationAgent,
)
from swarmauri.standard.conversations.concrete.Conversation import Conversation

# Fetch the API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

available_llms = {
    "GroqModel": (GroqModel, GROQ_API_KEY),
    "OpenAIModel": (OpenAIModel, OPENAI_API_KEY),
}

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

CHOSEN_LLM = available_llms["GroqModel"][0]
API_KEY = available_llms["GroqModel"][1]
DEFAULT_AI = "GroqModel"
DEFAULT_MODEL = llms["GroqModel"][0]
TEMPERATURE = 0.7
MAX_TOKENS = 512

# Store the generated code snippets globally
code_snippet = ""
new_code_snippet = ""


# Define the callback function for the LLM component dropdown
def llm_component_callback(component):
    if llms.get(component, None) is not None:
        global CHOSEN_LLM
        global API_KEY
        CHOSEN_LLM = available_llms[component][0]
        API_KEY = available_llms[component][1]
        return gr.update(choices=[model for model in llms[component]])
    return gr.update(choices=[])


# Define the callback function for the LLM model dropdown
def llm_model_callback(model):
    global new_code_snippet
    # Only update the model part of the code preview
    return f"Agent created with {model}"


# Define the function to handle conversation
def handle_conversation(llm_model, user_message, history, temperature, max_tokens):
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


# Define the function to generate the full code preview
def update_code_preview(component, model):
    api_key_mapping = {
        "GroqModel": "GROQ_API_KEY",
        "OpenAIModel": "OPENAI_API_KEY",
    }
    global code_snippet
    code_snippet = f"""
from swarmauri.standard.llms.concrete.{component} import {component} as LLM
from swarmauri.standard.conversations.concrete.Conversation import Conversation
from swarmauri.standard.messages.concrete.HumanMessage import HumanMessage

# model initialization
API_KEY = os.getenv('{api_key_mapping.get(component, None)}')
model = LLM(api_key=API_KEY, name='{model}')
conversation = Conversation()

# user input
input_data = "Hello"
human_message = HumanMessage(content=input_data)
conversation.add_message(human_message)

# prediction key word arguments
llm_kwargs = {{
    "temperature": 0.7,
    "max_tokens": 512,
}}

# messages
messages = {[f'HumanMessage(content={message.content})' if message.type=='HumanMessage' else f'HumanMessage(content={message.content})' for message in conversation.history]}

# prediction
model.predict(conversation=conversation, **llm_kwargs)
prediction = conversation.get_last().content
print(prediction)
"""
    return code_snippet


# Define the function to update the model in the code preview
def update_model_in_code_preview(model):
    global new_code_snippet
    new_code_snippet = code_snippet.replace(
        f"name='{DEFAULT_MODEL}'",  # Replace the default model
        f"name='{model}'",  # Replace with the new model
    )
    return new_code_snippet


# Initialize current values for temperature and max_tokens
current_temperature = TEMPERATURE
current_max_tokens = MAX_TOKENS


# Define the function to update both temperature and max_tokens in the code preview
def update_code_with_temperature_and_tokens(temperature, max_tokens):
    global new_code_snippet
    # Update both the temperature and max_tokens in the new_code_snippet
    updated_code_snippet = new_code_snippet.replace(
        f'"temperature": {TEMPERATURE}', f'"temperature": {temperature}'
    ).replace(f'"max_tokens": {MAX_TOKENS}', f'"max_tokens": {max_tokens}')

    # Update the global variables for the current temperature and max_tokens
    global current_temperature
    global current_max_tokens
    current_temperature = temperature
    current_max_tokens = max_tokens

    return updated_code_snippet


code_preview = None

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

        # Chat interface
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

        # Add a column for the code preview block without changing the order
        with gr.Column(scale=3):
            code_preview = gr.Code(
                language="python",
                label="Code Preview",
                value=update_code_preview(DEFAULT_AI, DEFAULT_MODEL),
            )

            # Set up the event to update the code preview when LLM component or model changes
            llm_component_dropdown.change(
                fn=lambda component: update_code_preview(
                    component, llm_model_dropdown.value
                ),
                inputs=llm_component_dropdown,
                outputs=code_preview,
            )

            llm_model_dropdown.change(
                fn=lambda model: update_model_in_code_preview(model),
                inputs=llm_model_dropdown,
                outputs=code_preview,
            )

            # Set up the event to update the temperature and max_tokens in the code preview
            temperature_slider.change(
                fn=lambda temperature: update_code_with_temperature_and_tokens(
                    temperature, current_max_tokens
                ),
                inputs=temperature_slider,
                outputs=code_preview,
            )

            max_tokens_slider.change(
                fn=lambda max_tokens: update_code_with_temperature_and_tokens(
                    current_temperature, max_tokens
                ),
                inputs=max_tokens_slider,
                outputs=code_preview,
            )

            # New event for user_input submission
            user_input.submit(
                fn=lambda: update_code_preview(
                    llm_component_dropdown.value, llm_model_dropdown.value
                ),
                inputs=[],  # No inputs here, using the dropdown values directly
                outputs=code_preview,
            )

# Run the interface
if __name__ == "__main__":
    interface.title = "Swarmauri Playground"
    interface.launch()
