import sys, os
import numpy as np 
import pandas as pd 
from PIL import Image
import json 
import cv2 
from torch.utils.data import Dataset  
import torchvision.transforms as T
import torch 

class OCID_dataset(Dataset):
	def __init__(self, root, annotation, transforms=None):
		'''
		annotation: a list of path that contain [csv, scene_list, class_idx]
		self.area -> the bounding box area
		'''
		self.root = root
		self.annotation = annotation
		self.transforms = transforms
		
		self.fname_path = []
		self.imgs = []
		with open(annotation[1], 'r') as f:
			for line in f: 
				path = os.path.join(self.root, line.split(' ')[1][:-1])
				self.fname_path.append(path)
				img = Image.open(path).convert('RGB')
				self.imgs.append(img)
		
		self.bbox = [[] for _ in range(len(self.imgs))]
		self.contour = [[] for _ in range(len(self.imgs))]
		self.labels = [[] for _ in range(len(self.imgs))]
		self.area = [[] for _ in range(len(self.imgs))]

		df = pd.read_csv(annotation[0])
		class_idx = json.load(open(annotation[2]))
		for _, row in df.iterrows():
			idx = row['scene_idx']
			bbox = json.loads(row['bbox'])
			self.bbox[idx].append(bbox)
			b_h, b_w = abs(float(bbox[2] - bbox[0])), abs(float(bbox[3] - bbox[1]))
			# self.area[idx].append(float(row['area'])) 
			self.area[idx].append(b_h * b_w) # use area of bbox
			self.labels[idx].append(class_idx[row['class']])
			seg = json.loads(row['segmentation'])	
			ct = np.array([[seg[i], seg[i+1]] for i in range(0, len(seg), 2)], dtype=np.int32)
			self.contour[idx].append(ct)
		
		self.mask = []
		for idx in range(len(self.imgs)):
			img = self.imgs[idx]
			w, h = img.size
			mask = []
			for ct in self.contour[idx]:
				m = np.zeros((h, w))
				# print(ct)
				cv2.fillPoly(m, [ct], color=1)
				mask.append(m.astype('uint8'))
			
			self.mask.append(mask)

	def __getitem__(self, idx):
		img = self.imgs[idx]

		target = {}
		target["boxes"] = torch.Tensor(self.bbox[idx])
		target["labels"] = torch.LongTensor(self.labels[idx])
		target["masks"] = torch.as_tensor(self.mask[idx], dtype=torch.uint8)
		target["image_id"] = torch.Tensor([idx])
		target["area"] = torch.Tensor(self.area[idx])
		target["iscrowd"] = torch.zeros((len(self.labels),), dtype=torch.int64)

		if self.transforms is not None:
			# img, target = self.transforms(img, target)
			img = self.transforms(img)

		return img, target
	
	def __len__(self):
		return len(self.imgs)

if __name__ == '__main__':
	data = OCID_dataset('data/OCID-dataset', ['data/annotation_train.csv', 'data/scene_train.txt', 'data/class_idx.json'], T.ToTensor())
	img, target = data.__getitem__(1)
	print(target)
