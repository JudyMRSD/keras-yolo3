# https://micropyramid.com/blog/building-and-parsing-xml-document-using-python/

# create an XML document
import xml.etree.ElementTree as xml

filename = "./output/test_xml.xml"
root = xml.Element("annotation")
userelement = xml.Element("user")
root.append(userelement)

uid = xml.SubElement(userelement, "uid")
uid.text = "1"

tree = xml.ElementTree(root)
with open(filename, "wb") as fh:
    tree.write(fh)
