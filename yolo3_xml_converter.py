# https://micropyramid.com/blog/building-and-parsing-xml-document-using-python/

# create an XML document
import xml.etree.ElementTree as xml
import cv2


class_name = "car"

folde_name = "frames_yolo_out"
filepath = './output/DJI_0005-0019.txt'
xml_out_folder = "./output_xml/"
filename = xml_out_folder + "DJI_0005-0019.xml"

fp = open(filepath)
# skip the first line (header)  image width height
line = fp.readline()
# with open(filepath) as fp:
line = fp.readline()
line = line.split(' ')
print('image=', line[0], 'height=',line[1],'width =', line[2], 'channels=', line[3])
img_filename, img_w, img_h,  img_channels = line[0], line[1], line[2], line[3]


cnt = 1
while line:
   line = fp.readline()
   line = line.split(' ')
   if len(line)<=1:
       print('finish file reading, invalid line')
       break
   print('class', line[0], 'xmin', line[1], 'ymin', line[2], 'xmax', line[3], 'ymax', line[4])
   bbox_class, xmin, ymin, xmax, ymax = line
   cnt += 1



root = xml.Element("annotation")

folder_tab = xml.SubElement(root, "folder")
folder_tab.text = folde_name

img_name_tab = xml.SubElement(root, "filename")
img_name_tab.text = img_filename


size_tab = xml.SubElement(root, "size")


size_width = xml.SubElement(size_tab, "width")
size_width.text = str(img_w)


size_height = xml.SubElement(size_tab, "height")
size_height.text = str(img_h)

size_channel = xml.SubElement(size_tab, "depth")
size_channel.text = str(img_channels)


object_tab = xml.SubElement(root, "object")
object_name = xml.SubElement(object_tab, "name")
object_name.text = class_name

bndbox_tab = xml.SubElement(object_tab, "bndbox")
bndbox_xmin = xml.SubElement(bndbox_tab, "xmin")
bndbox_xmin.text = xmin
bndbox_ymin = xml.SubElement(bndbox_tab, "ymin")
bndbox_ymin.text = ymin
bndbox_xmax = xml.SubElement(bndbox_tab, "xmax")
bndbox_xmax.text = xmax
bndbox_ymax = xml.SubElement(bndbox_tab, "ymax")
bndbox_ymax.text = ymax


tree = xml.ElementTree(root)
with open(filename, "wb") as fh:
    tree.write(fh)

