#iglob method for listing the files in a directory 
import glob 
path = ""
for file in glob.iglob(path, recursive=True):
  print(file)
