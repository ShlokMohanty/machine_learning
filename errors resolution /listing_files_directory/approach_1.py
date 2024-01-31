import os 
path = "" #in path to avoid the truncation error we the ("// ") should be used.
dir_list = os.listdir(path)
print("files and directories in '", path, "' :")
#print all the files 
print(dir_list)
