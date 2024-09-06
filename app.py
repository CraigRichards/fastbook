import gradio as gr
from PIL import Image
import numpy as np
from fastbook import *
from fastai.vision.widgets import *

path = Path()
path.ls(file_exts='.pkl')
learn_inf = load_learner(path/'export.pkl')
learn_inf.dls.vocab

# gradio app for the model
def classify_image(img):
    # Ensure img is a NumPy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    # Convert to uint8
    img = img.astype('uint8')
    img = Image.fromarray(img, 'RGB')
    img = img.resize((224, 224))

    pred, _, probs = learn_inf.predict(img)
    return {learn_inf.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

image = gr.components.Image(type='pil', label="Original Image")
label = gr.components.Label(num_top_classes=3)

# Correct paths to sample images
sample_images = [
    "images/grizzly.jpg",
    "images/black.jpg",
    "images/teddy.jpg"
]

# Load sample images as components
sample_image_components = [gr.components.Image(image_path) for image_path in sample_images]

image = gr.components.Image(type='pil', label="Original Image")
label = gr.components.Label(num_top_classes=3)
app = gr.Interface(
    fn=classify_image, 
    inputs=[image], 
    outputs=label, 
    examples=sample_images,
    title="Bear Classifier", 
    description="Identify bear species"
)


if __name__ == "__main__":
    app.launch(debug=True, server_port=7861)



