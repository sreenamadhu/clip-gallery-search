from PIL import Image
import numpy as np
import pandas as pd
import argparse
import clip
import torch
import requests
import io
from progressbar import progressbar
from torch.utils import data
from torch import nn

class Unsplash_Dataset(data.Dataset):
	def __init__(self, fname, transform):
		super(Unsplash_Dataset,self).__init__()
		self.fname = fname
		self.data_df = pd.read_csv(self.fname, sep = '\t', header = 0)
		self.num_samples = len(self.data_df)
		self.transform = transform
		self.file = open('images.txt','w+')

	def __getitem__(self,index):
		try:
			im_url = self.data_df['photo_image_url'][index]
			im = Image.open(io.BytesIO(requests.get(im_url).content))

			if self.transform is not None:
				im = self.transform(im)
			self.file.write('{}\n'.format(im_url))
			return im
		except:
			return None
	def __len__(self):

		return self.num_samples

def collate_fn(batch):
	batch = list(filter(lambda x: x is not None, batch))
	return torch.utils.data.dataloader.default_collate(batch)

def get_batch_im_features(model_image, dataloader):
	all_features = []
	with torch.no_grad():
		for images in progressbar(dataloader):
			features = model_image(images.to(device))
			all_features.append(features)

	return torch.cat(all_features)


'''
Model Setup
'''
class TextCLIP(nn.Module):
	def __init__(self, model) :
		super(TextCLIP, self).__init__()
		self.model = model
		
	def forward(self,text):
		return self.model.encode_text(text)
	
class ImageCLIP(nn.Module):
	def __init__(self, model) :
		super(ImageCLIP, self).__init__()
		self.model = model
		
	def forward(self,image):
		return self.model.encode_image(image)

		
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/32',device = device)
model.eval()
model_text = TextCLIP(model)
model_image = ImageCLIP(model)
model_text = torch.nn.DataParallel(model_text)
model_image = torch.nn.DataParallel(model_image)

'''
Extract image features
'''
dataset = Unsplash_Dataset('unsplash/photos.tsv000', transform = preprocess)
dataloader = data.DataLoader(dataset, batch_size=100, collate_fn=collate_fn)
image_features = get_batch_im_features(model_image, dataloader)
dataset.file.close()
torch.save(image_features,'unsplash/clip-ViT-B-32-unsplash-lite.pt')
