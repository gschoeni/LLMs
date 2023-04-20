import gradio as gr
import asyncio
import argparse

# from llm.models.tokenizers.flan_t5 import load_tokenizer
# from llm.models.llms.flan_t5 import inference, load_model

from llm.models.tokenizers.cerebras import load_tokenizer
from llm.models.llms.cerebras import inference, load_model

from oxen.repositories import Repository
from oxen.branches import Branch

async def main():
    parser = argparse.ArgumentParser(prog="UI for evaluating a LLM")
    parser.add_argument("-b", "--base_model", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("--host", default="hub.oxen.ai")
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--branch", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--share", default=False)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.base_model)
    model = load_model(args.model, tokenizer, device=args.device)

    async def predict(text, session_state, history=[]):
        print(f"Text: {text}")
        response = inference(model, tokenizer, text)
        history.append((text, response))

        last_message = {"prompt": text, "response": response}
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

    demo.launch(debug=True, server_name="0.0.0.0", share=args.share)


asyncio.run(main())
