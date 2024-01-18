import os
from tqdm import tqdm
import lzma

def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files
    
folder_path = "C:/Users/ishaa/Downloads/openwebtext/openwebtext"
#output_file = "output{}.txt"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"
#split_files = int(input("How many files?"))

files = xz_files_in_dir(folder_path)
total_files = len(files)

#max_count = total_files//split_files if split_files != 0 else total_files

# calculate the split:
split_index = int(total_files * 0.8) # 80% of files
files_train = files[:split_index]
files_val = files[split_index:]

# Processing training and Validation seperately:
vocab = set()

"""# processing:
for i in range(split_files):
    with open(output_file.format(i),"w", encoding="utf-8") as outfile:
        for count, filename in enumerate(tqdm(files[:max_count], total = max_count)):
            if count >= max_count:
                break
            file_path = os.path.join(folder_path, filename)
            with lzma.open(file_path, "rt", encoding = "utf-8") as infile:
                text = infile.read()
                outfile.write(text)
                characters = set(text)
                vocab.update(characters)
        files = files[max_count:]"""

# training files processing:
with open(output_file_train, "w", encoding = "utf-8") as outfile:
    for filename in tqdm(files_train, total = len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding = "utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)
            
# validation files processing:
with open(output_file_val, "w", encoding = "utf-8") as outfile:
    for filename in tqdm(files_val, total = len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding = "utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

with open(vocab_file, "w", encoding = "utf-8") as vfile:
    for char in vocab:
        vfile.write(char + '\n')
            