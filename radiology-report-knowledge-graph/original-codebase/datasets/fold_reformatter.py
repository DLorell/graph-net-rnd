import os


dataset_path = "datasets"


folds = [os.path.join("data", f) for f in os.listdir("data") if "fold" in f and "new" not in f]

for fold in folds:
    if os.path.isfile(fold+".new"):
        with open(fold+".new", 'r+') as fp:
            fp.truncate(0)


    with open(fold, 'r') as fp:
        with open(fold+".new", 'w+') as fp2:
            line = fp.readline()
            while line:
                new_line = line.strip().split(' ')[0]
                paths = line.strip().split(' ')[1:]
                new_paths = []
                for path in paths:
                    dirs = path.split('/')[2:]
                    new_path = dataset_path
                    for dir in dirs:
                        new_path = os.path.join(new_path, dir)
                    new_paths.append(new_path)

                for new_path in new_paths:
                    new_line += " " + new_path
                new_line += "\n"
                
                line = fp.readline()

                fp2.write(new_line)
