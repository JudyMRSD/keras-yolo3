import os
imgdir = "./input_imgs/imgs/2222/"
for filename in os.listdir(imgdir):
    newname = filename.replace("frame", "2222")
    os.rename(imgdir+filename, imgdir+newname)