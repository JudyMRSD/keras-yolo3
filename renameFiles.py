import os
imgdir = "./training_data/fifth_dataset/images_may4/"
for filename in os.listdir(imgdir):
    newname = filename.replace('rotatedroadframe', '')
    os.rename(imgdir+filename, imgdir+newname)