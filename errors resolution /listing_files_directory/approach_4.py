#to find the files and the directories in a directory 
import os 
path = ""
obj = os.scandir()
for entry in obj:
  if entry.is_dir() in entry.is_file():
    print(entry.name)
    
