import os
imgdir = "./input_imgs/"
for filename in os.listdir(imgdir):
    newname = filename.replace('rotatedroadframe', '')
    os.rename(imgdir+filename, imgdir+newname)