import sys, os
import copy
from PIL import Image
import argparse
import numpy as np
import torch
import json 
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as T
import h5py

from data_loader import OCID_dataset
from engine import train_one_epoch, evaluate
import utils

def set_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help="Training the model from pretrained weight", action='store_true', default=False)
	parser.add_argument('--test', help="Testing the model", action='store_true', default=False)
	parser.add_argument('--pred', help="Predict and get the input of DGA", action='store_true', default=False)
	
	return parser.parse_args()

def self_testing(model):
	torch.set_printoptions(sci_mode=False)
	img = Image.open('src/test.png')
	img = ToTensor()(img).unsqueeze(0)
	
	model.eval()
	res = (model.cuda())(img.cuda())

	v = res[0]['masks'][1].squeeze(0).T
	for row in v: 
		if torch.sum(row) > 20:
			for i in row:
				if i > 0.5: 
					print(i)
			exit()

def get_model(class_num):
	model = maskrcnn_resnet50_fpn(pretrained=True)
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, class_num)

	in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
	hidden_layer = 256
	model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, class_num)

	# print(model)
	return model 

def train(model, data_train, data_val, epochs=1000, model_path='model/mask_rcnn_1101.pt'):
	device = torch.device('cuda')
	train_loader = DataLoader(data_train, batch_size=16, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
	val_loader = DataLoader(data_val, batch_size=16, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
	
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.00005)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
	
	max_map = -1
	not_improve = 0
	for e in range(epochs):
		train_one_epoch(model, optimizer, train_loader, device, epochs, print_freq=15)
		lr_scheduler.step()
		print('--------------- Training performance ---------------')
		evaluate(model, train_loader, device=device)
		print('--------------- Validation performance ---------------')
		evaluator = evaluate(model, val_loader, device=device)
		print('---------------------------------------------------')

		map05 = evaluator.coco_eval['segm'].stats[1]
		if map05 > max_map:
			torch.save(model.state_dict(), model_path)
			max_map = map05
			not_improve = 0
		else: 
			not_improve += 1

		if not_improve >= 10:
			print(f'Best mAP0.5 = {max_map}')
			return 

def test(data_train, data_val, model_path='model/mask_rcnn_1101.pt'):
	device = torch.device('cuda')
	model = get_model(class_num).cuda()
	model.load_state_dict(torch.load(model_path))
	model.eval()

	train_loader = DataLoader(data_train, batch_size=16, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
	val_loader = DataLoader(data_val, batch_size=16, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
	print('--------------- Final training performance ---------------')
	evaluate(model, train_loader, device=device)
	print('--------------- Final validation performance ---------------')
	evaluator = evaluate(model, val_loader, device=device)

def predict(model, data, opt_root):
	model.eval()
	loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)
	
	for idx, (img, _) in enumerate(loader):
		img = img.cuda()
		opt = model(img)
		print(fc7_out.shape)
		exit()
		fname = data.fname_path[idx]
		os.makedirs(os.path.join(opt_root, '/'.join(fname.split('/')[2:-1])), exist_ok=True)
		opt_path = os.path.join(opt_root, '/'.join(fname.split('/')[2:]))[:-4] + '.hdf5'

		with h5py.File(opt_path, 'w') as f:
			f.create_dataset('fc7', data=fc7_out)
			f.create_dataset('obj_num', data=len(opt))
			
			for i, obj in enumerate(opt):
				grp = f.create_group('objs')
				grp.create_dataset('boxes', data=obj['boxes'].cpu().detach())
				grp.create_dataset('labels', data=obj['labels'].cpu().detach())
				grp.create_dataset('scores', data=obj['scores'].cpu().detach())
				grp.create_dataset('masks', data=obj['masks'].cpu().detach())

		print(f'Finished {idx+1}/{len(loader)}  ', end = '\r')

fc7_out = -1
def hook(module, fea_in, fea_out):
	global fc7_out
	fc7_out = fea_out.cpu().detach()
	
	return None

if __name__ == '__main__':
	class_num = 59 # bg + 23/58
	model_path='model/mask_rcnn_1101.pt'
	args = set_args()

	model = get_model(class_num)

	trans_train = T.Compose([T.ToTensor(), T.ColorJitter(0.2, 0.2, 0.2), T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	trans_val = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

	
	if args.train:
		data_train = OCID_dataset('data/OCID-dataset', ['data/annotation_train.csv', 'data/scene_train.txt', 'data/class_idx.json'], trans_train)
		print('Finish loading training set')
		data_val = OCID_dataset('data/OCID-dataset', ['data/annotation_val.csv', 'data/scene_val.txt', 'data/class_idx.json'], trans_val)
		print('Finish loading validation set')

		model = get_model(class_num).cuda()
		model = train(model, data_train, data_val, model_path=model_path)
	
	if args.test:
		data_train = OCID_dataset('data/OCID-dataset', ['data/annotation_train.csv', 'data/scene_train.txt', 'data/class_idx.json'], trans_train)
		data_val = OCID_dataset('data/OCID-dataset', ['data/annotation_val.csv', 'data/scene_val.txt', 'data/class_idx.json'], trans_val)
		print('Finish loading validation set')
	
		test(data_train, data_val, model_path)
	
	if args.pred: 
		data_train = OCID_dataset('data/OCID-dataset', ['data/annotation_train.csv', 'data/scene_train.txt', 'data/class_idx.json'], trans_val)
		print('Finish loading training set')
		data_val = OCID_dataset('data/OCID-dataset', ['data/annotation_val.csv', 'data/scene_val.txt', 'data/class_idx.json'], trans_val)
		print('Finish loading validation set')

		model = get_model(class_num).cuda()
		model.load_state_dict(torch.load(model_path))
		model.eval()

		model.roi_heads.box_head.fc7.register_forward_hook(hook)
		
		predict(model, data_train, opt_root='DGA_input')

		
	

	
