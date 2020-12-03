import pandas as pd 
import json 

df = pd.read_csv('data/annotation_new.csv')

rec_label = {}
for _, label in df.iterrows():
	l = label['class']
	if l not in rec_label:
		rec_label[l] = len(rec_label) + 1

with open('data/class_idx.json', 'w') as f:
	json.dump(rec_label, f)
