from PIL import Image
import numpy as np
import argparse
import clip
import torch
import requests
import io
from progressbar import progressbar
from torch import nn
import os

def image_to_image_search(model, image_feature,image_features):
	image_features /= image_features.norm(dim=-1, keepdim=True)
	image_feature /= image_feature.norm(dim=-1, keepdim=True)
	similarity = (100.0 * image_features @ image_feature.T)
	similarity = similarity.softmax(dim=0)
	values, indices = similarity.T.topk(args.topk)
	return values[0].cpu().numpy(), indices[0].cpu().numpy()

def export_topk(output_folder,values, indices):
	urls = [x.strip('\n') for x in open('unsplash/image_urls.txt','r').readlines()]
	for k, (val,idx) in enumerate(zip(values, indices)):
		im_url = urls[idx]
		im = Image.open(io.BytesIO(requests.get(im_url).content))
		im.save(os.path.join(output_folder,'top{}_conf{}.jpg'.format(k,val)))
	print('Results saved at {}'.format(output_folder))
	return



if __name__ == '__main__':

	parser = argparse.ArgumentParser(
	        description="Image to Image Search")
	parser.add_argument('-i','--test_img', type = str, required = True, help = 'Input text to search the unsplash dataset')
	parser.add_argument('-o', '--output_folder', type = str, default = 'image_image_output/', help = 'Folder to output the matched images')
	parser.add_argument('-k', '--topk', type = int, default = 10, help = 'Number of matched images to output')                        
	args = parser.parse_args()

	'''
	Model Setup
	'''
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model, preprocess = clip.load('ViT-B/32',device = device)
	model.eval()


	#Load Image features
	image_features = torch.load('unsplash/clip-ViT-B-32-unsplash-lite.pt')

	#Extract text feature
	im = Image.open(args.test_img)
	im = preprocess(im).to(device).unsqueeze(0)
	with torch.no_grad():
		print(im.shape)
		image_feature = model.encode_image(im)

	#Text to Image Matching
	values, indices = image_to_image_search(model, image_feature,image_features)

	# Export results
	export_path = os.path.join(args.output_folder, args.test_img.split('.')[0])
	if not os.path.isdir(export_path):
		os.makedirs(export_path)
	export_topk(export_path, values, indices)

