import os
from shutil import copy2

folds = [os.path.join("data", f) for f in os.listdir("data") if "fold" in f]

splits = {"front": [], "left": [], "right": []}

for fold in folds:
    with open(fold, 'r') as fp:
        line = fp.readline()
        while line:
            pieces = line.strip().split(' ')[1:]
            for piece in pieces:
                atoms = piece.split('/')[-2:]
                place = atoms[0]
                name = atoms[1]
                
                assert place in splits.keys()
                splits[place].append(name)
            line = fp.readline()


for place in ["front", "left", "right"]:
    for f in splits[place]:
        copy2("datasets/og_data/"+f, "datasets/"+place)


