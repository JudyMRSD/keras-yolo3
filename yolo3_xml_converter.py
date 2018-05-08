# https://micropyramid.com/blog/building-and-parsing-xml-document-using-python/

# create an XML document
import xml.etree.ElementTree as xml
import cv2
folde_name = "frames_yolo_out"
img_filename ="300.jpg"
img = cv2.imread(img_filename)
img.shape()

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

