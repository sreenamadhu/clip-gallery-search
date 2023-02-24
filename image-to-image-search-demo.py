from PIL import Image
import numpy as np
import argparse
import clip
import torch
import requests
import io
#from progressbar import progressbar
from torch import nn
import os
import gradio as gr

'''
Model Setup
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model, preprocess = clip.load('ViT-B/32',device = device)
model.eval()

#Load Image features
dataset_features = torch.load('unsplash/clip-ViT-B-32-unsplash-lite.pt', map_location = device)
dataset_features /= dataset_features.norm(dim=-1, keepdim = True)
urls = [x.strip('\n') for x in open('unsplash/image_urls.txt','r').readlines()]


def image_to_image_search(image, topk):
	topk = int(topk)
	image = preprocess(image).to(device).unsqueeze(0)
	with torch.no_grad():
		image_feature = model.encode_image(image)
	image_feature /= image_feature.norm(dim=-1, keepdim=True)
	similarity = (100.0*dataset_features@image_feature.T)
	similarity = similarity.softmax(dim=0)
	values, indices = similarity.T.topk(topk)
	values, indices = values[0].cpu().numpy(), indices[0].cpu().numpy()
	out = []
	for k, (val,idx) in enumerate(zip(values, indices)):
		im_url = urls[idx]
		im = Image.open(io.BytesIO(requests.get(im_url).content))
		out.append(im)
	return out

with gr.Blocks() as demo:
	gr.Markdown("<h1><center> Image Search Demo</h1>")
	gr.Markdown("<center> This tool is used to run the search based on the image.")
	with gr.Row():
		with gr.Column():
			with gr.Row():
				input_image = gr.Image(label='Query Image', type = 'pil')
			with gr.Row():
				slider = gr.Slider(label = 'Top K', minimum=1, maximum=25, value = 10, step = 1)
			with gr.Row():
				submit = gr.Button('Submit', variant = 'primary')
				clear = gr.Button('Clear', variant = 'secondary')
			inputs = [input_image, slider]
		with gr.Column():
			outputs = gr.Gallery(label = "Top Matches").style(grid = [4], height = 300, container = True)

	with gr.Row():
		with gr.Column():
			gr.Examples(
				examples=[["beach.jpg",15], ["dog.jpg",15]],
				inputs = inputs,)

	submit.click(image_to_image_search, inputs, outputs)
	clear.click(lambda x:None, input_image, input_image)
demo.launch(server_port = 8082)

