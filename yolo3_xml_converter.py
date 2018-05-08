# https://micropyramid.com/blog/building-and-parsing-xml-document-using-python/

# create an XML document
import xml.etree.ElementTree as xml
import cv2


filepath = './output/DJI_0005-0019.txt'

fp = open(filepath)
# skip the first line (header)  image width height
line = fp.readline()
# with open(filepath) as fp:
line = fp.readline()
line = line.split(' ')
print('image=', line[0], 'height=',line[1],'width =', line[2])
img_filename, img_w, img_h = line[0], line[1], line[2]


cnt = 1
while line:
   line = fp.readline()
   line = line.split(' ')
   if len(line)<=1:
       print('finish file reading, invalid line')
       break
   print('class', line[0], 'xmin', line[1], 'ymin', line[2], 'xmax', line[3], 'ymax', line[4])
   cnt += 1



folde_name = "frames_yolo_out"

filename = "./output/test_xml.xml"
root = xml.Element("annotation")

folder_tab = xml.SubElement(root, "folder")
folder_tab.text = folde_name

img_name_tab = xml.SubElement(root, "filename")
img_name_tab.text = img_filename


size_tab = xml.SubElement(root, "size")
root.append(size_tab)

size_width = xml.SubElement(size_tab, "width")
size_width.text = str(img_w)



tree = xml.ElementTree(root)
with open(filename, "wb") as fh:
    tree.write(fh)

