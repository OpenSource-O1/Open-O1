import gradio as gr
from loguru import logger
from V3 import call_gpt

class Conversation:
    def __init__(self, max_history_len=10):
        self.max_history_len = max_history_len

    def pop_history(self, history):
        while len(history) > self.max_history_len:
            for item in history:
                if item["role"] == "user":
                    history.remove(item)
                    break
            for item in history:
                if item["role"] == "assistant":
                    history.remove(item)
                    break
        return history

    def ask(self, history, prompt):
        history = self.pop_history(history)
        logger.info(history)
        return call_gpt(history, prompt)

conv = Conversation()

def make_history(system_prompt, qa_list):
    history = [{"role": "system", "content": system_prompt}]
    for q, a in qa_list:
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": a})
    return history

def answer(system_prompt, prompt, history=[]):
    history.append(prompt)
    qa_list = [(u, b) for u, b in zip(history[::2], history[1::2])]
    message = conv.ask(make_history(system_prompt, qa_list), prompt)
    
    # 对反引号进行转义
    message = message.replace("`", "\\`")
    
    # 包裹为代码块
    message = f"```\n{message}\n```"
    
    history.append(message)

    chatbot_messages = []
    for q, a in qa_list:
        chatbot_messages.append((q, a))
    
    chatbot_messages.append((prompt, message))
    
    return "", chatbot_messages, history

def clear_history(state):
    state.clear()
    return state, []

with gr.Blocks(css="#chatbot{height:500px} .overflow-y-auto{height:500px}") as rxbot:
    with gr.Row():
        sys = gr.Textbox(show_label=False, value="You are open-o1, a helpful assistant.")
    chatbot = gr.Chatbot()
    state = gr.State([])
    
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="请输入你的问题", max_lines=8)
        
    with gr.Row():
        clear_button = gr.Button("🧹Clear History")
        send_button = gr.Button("🚀Send")
        
        send_button.click(
            fn=answer,
            inputs=[sys, txt, state],
            outputs=[txt, chatbot, state]
        )

        txt.submit(
            fn=answer,
            inputs=[sys, txt, state],
            outputs=[txt, chatbot, state]
        )

        clear_button.click(
            fn=clear_history,
            inputs=[state],
            outputs=[state, chatbot]
        )

rxbot.launch()
