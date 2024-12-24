from langchain.memory import ChatMessageHistory
import pprint
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)


chat_message_history = ChatMessageHistory()

def add_user_message_from_prompt(prompt, key=None):
    if key is None:
        prompt_str = prompt
    else:
        prompt_str = prompt[key]
    chat_message_history.add_user_message(prompt_str)

    pprint.pprint("User Message: \n" +  prompt_str)  # Print user message in a beautiful way

    return prompt

def add_ai_message_from_output(output, name):
    # import pdb; pdb.set_trace()
    if type(output) == str:
        chat_message_history.add_ai_message(AIMessage(content=output, name=name))
    else:
        chat_message_history.add_ai_message(AIMessage(content=str(output), name=name))
    
    pprint.pprint("AI Message: \n" + str(output))  # Print AI message in a beautiful way

    return output