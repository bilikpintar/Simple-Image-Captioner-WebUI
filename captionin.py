import gradio as gr
import google.generativeai as genai
import os
from dotenv import load_dotenv
import PIL.Image
from PIL import Image

load_dotenv()

# Set the API key in the environment variable
os.environ["GOOGLE_API_KEY"] = "YOUR GEMINI API TOKEN"

# Configure the API key for Google Generative AI
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel("gemini-pro-vision")

def resize_image(input_image_path, max_size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    print(f"The original image size is {width} wide x {height} tall")

    # Calculate the ratio
    ratio = min(max_size/width, max_size/height)
    new_size = (int(width*ratio), int(height*ratio))

    resized_image = original_image.resize(new_size, Image.LANCZOS)
    width, height = resized_image.size
    print(f"The resized image size is {width} wide x {height} tall")
    return resized_image  

def process_image(img_path):
    print("Image Path:", img_path)
    prompt = "Describe the persons, The person is appearance like eyes color, hair color, skin color, camera angle, and the clothes, object position the scene and the situation. Please describe it detailed. Don't explain the artstyle of the image"
    print("Prompt:", prompt)
    
    # Resize the image before processing
    img = resize_image(img_path, 2048)  # Adjust the size as needed
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()
    return refine_caption(response.text)

def refine_caption(caption):
    # Remove unnecessary phrases
    phrases_to_remove = [
        "The image shows", "The image is", "looking directly at the camera",
        "in the image", "taking a selfie", "posing for a picture",
        "holding a cellphone", "is wearing a pair of sunglasses",
        "pulled back in a ponytail", "with a large window in the cent", "The picture shows", "The person is", "This is a picture of", "..", "There are"
    ]
    for phrase in phrases_to_remove:
        caption = caption.replace(phrase, "")

    # Clean up the caption
    caption = caption.replace("..", ".").replace(" and.", "").replace(" is.", "").strip()

    # Limit to the first 50 sentences
    sentences = caption.split('. ')
    sentences = [s.strip() + '.' for s in sentences[:50]]
    return ' '.join(sentences)

iface = gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="filepath")],
    outputs="text",
    live=False,
)

iface.queue()
iface.launch(server_name="0.0.0.0", server_port=7866)
