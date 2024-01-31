#accessing files in the directory tree 
import os 
path = ""
list = []
for(root, dirs, file) in os.walk(path):
  for f in file:
    if '.xlsx' in f:
      print(f)
  
    
