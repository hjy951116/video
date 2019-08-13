
import os
def get_fileNames(rootdir):    
	fs = []    
	for root, dirs, files in os.walk(rootdir,topdown = True):
		for name in files: 
			_, ending = os.path.splitext(name)
				if ending == ".jpg":
					fs.append(os.path.join(root,name))   
	return fs
