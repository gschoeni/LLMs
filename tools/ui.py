import gradio as gr
import asyncio
import argparse
import time

from llm.models.llms.factory import get_model

from oxen.repositories import Repository
from oxen.branches import Branch

async def main():
    parser = argparse.ArgumentParser(prog="UI for evaluating a LLM")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("--host", default="hub.oxen.ai")
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--branch", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--share", default=False)
    args = parser.parse_args()

    model = get_model(args.model)
    
#     instruction = """"Assistant is a large language model trained by OpenAI.

# Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

# Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

# Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
# """

    instruction = ""

    async def predict(text, session_state, history=[]):
        print(f"Text: {text}")
        
        # prompt = instruction + "\nHuman: " + text + "\nAssistant: "
        prompt = text

        start = time.time()
        response = model(prompt)
        end = time.time()
        elapsed = end - start

        history.append((text, response))

        last_message = {"instruction": instruction, "input": text, "prompt": prompt, "response": response, "model": args.model, "response_time": elapsed}
        print(f"last message: {last_message}")
        
        session_state["history"] = history
        session_state["last_message"] = last_message

        return history, gr.Textbox.update(value="")

    def maybe_save_row(row: dict):
        if args.repo is not None:
            file = args.file
            repo = Repository(args.repo, host=args.host)
            branch = repo.get_branch_by_name(args.branch)
            repo.add_row(branch, file, row)

    async def upvote_response(session_state):
        print("upvote_response")
        history = session_state["history"]
        
        last_message = session_state["last_message"]
        last_message["upvote"] = True
        maybe_save_row(last_message)

        return history, gr.Textbox.update(value="")

    async def downvote_response(session_state):
        print("downvote_response")
        history = session_state["history"]

        last_message = session_state["last_message"]
        last_message["upvote"] = False
        maybe_save_row(last_message)
        
        if len(history) > 0:
            last_interaction = history.pop()
            print(last_interaction)

        return history, gr.Textbox.update(value="")
    
    async def reset_history(session_state):
        print("reset_history")
        session_state["history"] = []
        return [], gr.Textbox.update(value="")

    with gr.Blocks(css="{max-width: 400px, background-color: red}") as demo:
        session_state = gr.State({})
        with gr.Row():
            gr.Markdown("# OxenAI ChatBot Demo")
        with gr.Row():
            chat_box = gr.Chatbot()
        with gr.Row():
            text_box = gr.Textbox(placeholder="Ask a question")
        with gr.Row():
            predict_btn = gr.Button("Reply", variant="primary")
            predict_btn.click(
                fn=predict,
                inputs=[text_box, session_state],
                outputs=[chat_box, text_box],
            )
        with gr.Row():
            with gr.Column():
                flag_btn = gr.Button("👍")
                flag_btn.click(
                    fn=upvote_response,
                    inputs=[session_state],
                    outputs=[chat_box, text_box],
                )
            with gr.Column():
                flag_btn = gr.Button("👎")
                flag_btn.click(
                    fn=downvote_response,
                    inputs=[session_state],
                    outputs=[chat_box, text_box],
                )
        with gr.Row():
            flag_btn = gr.Button("Reset")
            flag_btn.click(
                fn=reset_history,
                inputs=[session_state],
                outputs=[chat_box, text_box],
            )

    demo.launch(debug=True, server_name="0.0.0.0", share=args.share)


asyncio.run(main())

# TODO: can we load data from a dataframe, and use keyboard shortcuts to navigate and annotate?