import gradio as gr
def process(x):
    print(type(x))
    if isinstance(x, dict):
        print(x.keys())
    return None

gr.ImageEditor(type="pil")
