import pandas as pd 
import numpy as np 
import os, sys 

def method1():
	sep_num = 1820
	with open('data/scene_list.txt', 'r') as f:
		f_train = open('data/scene_train.txt', 'w')
		f_val = open('data/scene_val.txt', 'w')
			
		for line in f:
			idx, path = line.split(' ')
			idx = int(idx)
			if idx < sep_num:
				f_train.write(f'{idx} {path}')
			else:
				f_val.write(f'{idx-sep_num} {path}')

	with open('data/annotation_new.csv', 'r') as f:
		f_train = open('data/annotation_train.csv', 'w')
		f_val = open('data/annotation_val.csv', 'w')

		for line in f:
			if line[0] == 's':
				f_train.write(line)
				f_val.write(line)
				continue
			l = line.split(',')
			idx = int(l[0])
			if idx < sep_num:	
				f_train.write(f"{idx},{','.join(l[1:])}")
			else:
				f_val.write(f"{idx-sep_num},{','.join(l[1:])}")

def method2():
	'''
	Validation policy:
		ARID10: 31, 33, 35, 37, 39 -> 12.8%
		ARID20: 12, 13 -> 15.4%
		YCB10: 3, 11, 25 -> 12.5%
	'''
	train_id = {}
	val_id = {}
	with open('data/scene_list.txt', 'r') as f:
		f_train = open('data/scene_train.txt', 'w')
		f_val = open('data/scene_val.txt', 'w')
		
		for line in f:
			idx = line.split(' ')[0]
			genre = line.split(' ')[1].split('/')[0]
			seq_num = int(line.split('seq')[1][:2])
			if genre == 'ARID10':
				if seq_num > 30 and seq_num % 2:
					val_id[idx] = len(val_id)
					f_val.write(f"{val_id[idx]} {line.split(' ')[1]}")
				else:
					train_id[idx] = len(train_id)
					f_train.write(f"{train_id[idx]} {line.split(' ')[1]}")
			elif genre == 'ARID20':
				if seq_num == 12 or seq_num == 13:
					val_id[idx] = len(val_id)
					f_val.write(f"{val_id[idx]} {line.split(' ')[1]}")
				else:
					train_id[idx] = len(train_id)
					f_train.write(f"{train_id[idx]} {line.split(' ')[1]}")
			elif genre == 'YCB10':
				if seq_num == 3 or seq_num == 25 or seq_num == 11:
					val_id[idx] = len(val_id)
					f_val.write(f"{val_id[idx]} {line.split(' ')[1]}")
				else:
					train_id[idx] = len(train_id)
					f_train.write(f"{train_id[idx]} {line.split(' ')[1]}")
			else:
				print('genre error')
				exit()
	
	with open('data/annotation_new.csv', 'r') as f:
		f_train = open('data/annotation_train.csv', 'w')
		f_val = open('data/annotation_val.csv', 'w')
		
		for line in f:
			idx = line.split(',')[0]
			if idx[0] == 's':
				f_train.write(line)
				f_val.write(line)
			else:
				content = ','.join(line.split(',')[1:])
				if idx in train_id:
					f_train.write(f'{train_id[idx]},{content}')
				else:
					f_val.write(f'{val_id[idx]},{content}')
		

def record():
	idx_dict = {'ARID10': 0, 'ARID20': 1, 'YCB10': 2}
	rec = [{}, {}, {}]
	total = [0, 0, 0]
	with open('data/scene_list.txt') as f:
		for line in f:
			idx = idx_dict[line.split(' ')[1].split('/')[0]]
			seq = line.split('seq')[1][:2]
			if seq in rec[idx]:
				rec[idx][seq] += 1;
			else:
				rec[idx][seq] = 1;
			total[idx] += 1;

	for l, t in zip(rec, total):
		print(l, t)

if __name__ == '__main__':
	record()
	method2()
