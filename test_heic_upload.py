import gradio as gr
from PIL import Image
from pillow_heif import register_heif_opener
import os

register_heif_opener()

with gr.Blocks() as app:
    f = gr.File(label="Upload file")
    img = gr.ImageEditor(label="Editor")
    
    def convert(file_obj):
        if file_obj is None: return None
        # Try opening
        try:
            return Image.open(file_obj.name)
        except Exception as e:
            print("Error:", e)
            return None
            
    f.change(convert, inputs=f, outputs=img)
