#getting all the files and no folders 
import os 
Dir = input()
files= os.listdir(Dir)
files = [f for f in files if os.path.isfile(Dir+'/'+f)]
print(*files, sep="\n")
