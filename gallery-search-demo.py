from PIL import Image
import numpy as np
import argparse
import clip
import torch
import requests
import io
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


def text_to_image_search(text, topk):
	topk = int(topk)
	#Extract text feature
	text_input = clip.tokenize(text).to(device)
	with torch.no_grad():
		text_feature = model.encode_text(text_input)
	text_feature /= text_feature.norm(dim=-1, keepdim=True)
	similarity = (100.0 * dataset_features @ text_feature.T)
	similarity = similarity.softmax(dim=0)
	values, indices = similarity.T.topk(topk)
	values, indices = values[0].cpu().numpy(), indices[0].cpu().numpy()
	out = []
	for k, (val,idx) in enumerate(zip(values, indices)):
		im_url = urls[idx]
		im = Image.open(io.BytesIO(requests.get(im_url).content))
		out.append(im)
	return out


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
	gr.Markdown("<h1><center> Gallery Search Demo</h1>")
	gr.Markdown("<center> This tool is used to run the search based on the image or text query.")
	with gr.Row():
		with gr.Column():
			with gr.Tab('Image Search') as img:
				with gr.Row():
					input_image = gr.Image(label='Query Image', type = 'pil')
				with gr.Row():
					slider = gr.Slider(label = 'Top K', minimum=1, maximum=25, value = 10, step = 1)
				with gr.Row():
					submit_img = gr.Button('Submit', variant = 'primary')
					clear_img = gr.Button('Clear', variant = 'secondary')
				inputs_img = [input_image, slider]
			with gr.Tab('Text Search') as txt:
				with gr.Row():
					input_txt = gr.Textbox(label='Query Text')
				with gr.Row():
					slider = gr.Slider(label = 'Top K', minimum=1, maximum=25, value = 10, step = 1)
				with gr.Row():
					submit_txt = gr.Button('Submit', variant = 'primary')
					clear_txt = gr.Button('Clear', variant = 'secondary')
				inputs_txt = [input_txt, slider]

		with gr.Column():
			outputs = gr.Gallery(label = "Top Matches").style(grid = [4], height = 300, container = True)

	with gr.Row():
		with gr.Column():
			with gr.Tab('Image Examples'):
				gr.Examples(
					examples=[["beach.jpg",15], ["dog.jpg",15]],
					inputs = inputs_img,)
			with gr.Tab('Text Examples'):
				gr.Examples(
					examples=[["beautiful smile",15], ["a picture of nature", 15]],
					inputs = inputs_txt)


	submit_img.click(image_to_image_search, inputs_img, outputs)
	clear_img.click(lambda x:None, input_image, input_image)

	submit_txt.click(text_to_image_search, inputs_txt, outputs)
	clear_txt.click(lambda x:"", input_txt, input_txt)
demo.launch(server_port = 8082)

