import gradio as gr
import asyncio
import argparse

from llm.models.tokenizers.cerebras import load_tokenizer
from llm.models.llms.cerebras import inference, load_model


async def main():
    parser = argparse.ArgumentParser(prog="UI for evaluating a LLM")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("--share", default=False)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, device=args.device)

    async def predict(text, session_state, history=[]):
        print(f"Text: {text}")
        response = inference(model, tokenizer, text, device=args.device)
        history.append((text, response))
        print(f"History in gradio:")
        for (i, o) in history:
            print(f"Human: {i}")
            print(f"AI: {o}")

        session_state["history"] = history

        return history, gr.Textbox.update(value="")

    async def upvote_response(session_state):
        print("upvote_response")
        history = session_state["history"]
        return history, gr.Textbox.update(value="")

    async def downvote_response(session_state):
        print("downvote_response")
        history = session_state["history"]

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
