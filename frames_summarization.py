import os
import openai
import base64
from PIL import Image

openai.api_key = "YOUR_OPENAI_API_KEY"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def gpt4v_describe_image(image_b64):
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": "You are an assistant skilled at describing the visual content of images. Your descriptions should be short, accurate, and written in natural English."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please describe the main objects and scene in this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=150
    )
    return response.choices[0].message["content"].strip()

def summarize_all_descriptions(descriptions):
    joined = "\n".join(f"- {desc}" for desc in descriptions)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes video content based on visual scene descriptions."},
            {"role": "user", "content": f"The following are descriptions of keyframes from a video:\n{joined}\nPlease summarize the overall content and theme of the video in 2–3 sentences."}
        ],
        max_tokens=200
    )
    return response.choices[0].message["content"].strip()

def describe_keyframes_from_folder(folder_path):
    keyframe_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])
    descriptions = []

    for fname in keyframe_files:
        print(f"Processing {fname} ...")
        img_b64 = encode_image_to_base64(os.path.join(folder_path, fname))
        desc = gpt4v_describe_image(img_b64)
        print(f"Description: {desc}")
        descriptions.append(desc)

    print("\nGenerating overall video summary...")
    overall_summary = summarize_all_descriptions(descriptions)

    print("\nKeyframe Descriptions:")
    for i, d in enumerate(descriptions):
        print(f"Frame {i + 1}: {d}")

    print("\nOverall Video Summary:")
    print(overall_summary)

    return descriptions, overall_summary