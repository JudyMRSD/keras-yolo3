# https://micropyramid.com/blog/building-and-parsing-xml-document-using-python/

# create an XML document
import xml.etree.ElementTree as xml
import cv2
import glob
import os


class Txt_To_Xml:
    def __init__(self, class_name):
        self.class_name = class_name
    def parse_txt(self, txt_folder, txtpath, xml_out_folder):
        self.txt_folder = txt_folder
        self.txtpath = txtpath
        base = os.path.basename(txtpath)
        base = os.path.splitext(base)[0]
        self.xml_path = xml_out_folder + base + ".xml"
        f_txt = open(self.txtpath)
        # skip the first line (header)  image width height
        line = f_txt.readline()
        # with open(filepath) as fp:
        line = f_txt.readline()
        line = line.split(' ')
        print('image=', line[0], 'height=',line[1],'width =', line[2], 'channels=', line[3])
        self.img_filename, self.img_w, self.img_h,  self.img_channels = line[0], line[1], line[2], line[3]

        self.img_info()

        cnt = 1
        while line:
           line = f_txt.readline()
           line = line.split(' ')
           if len(line)<=1:
               print('finish file reading, invalid line')
               break
           print('class', line[0], 'xmin', line[1], 'ymin', line[2], 'xmax', line[3], 'ymax', line[4])
           _, self.xmin, self.ymin, self.xmax, self.ymax = line
           self.bbox_info()
           cnt += 1

        tree = xml.ElementTree(self.root)
        with open(self.xml_path, "wb") as fh:
            tree.write(fh)


    def img_info(self):
        self.root = xml.Element("annotation")

        folder_tab = xml.SubElement(self.root , "folder")
        folder_tab.text = self.txt_folder

        img_name_tab = xml.SubElement(self.root , "filename")
        img_name_tab.text = self.img_filename


        size_tab = xml.SubElement(self.root , "size")


        size_width = xml.SubElement(size_tab, "width")
        size_width.text = str(self.img_w)


        size_height = xml.SubElement(size_tab, "height")
        size_height.text = str(self.img_h)

        size_channel = xml.SubElement(size_tab, "depth")
        size_channel.text = str(self.img_channels)

    def bbox_info(self):

        object_tab = xml.SubElement(self.root, "object")
        object_name = xml.SubElement(object_tab, "name")
        object_name.text = self.class_name

        bndbox_tab = xml.SubElement(object_tab, "bndbox")
        bndbox_xmin = xml.SubElement(bndbox_tab, "xmin")
        bndbox_xmin.text = self.xmin
        bndbox_ymin = xml.SubElement(bndbox_tab, "ymin")
        bndbox_ymin.text = self.ymin
        bndbox_xmax = xml.SubElement(bndbox_tab, "xmax")
        bndbox_xmax.text = self.xmax
        bndbox_ymax = xml.SubElement(bndbox_tab, "ymax")
        bndbox_ymax.text = self.ymax


def main():
    class_name = "car"
    txt_folder = "frames_yolo_out"
    # txtpath = './output/DJI_0005-0019.txt'
    xml_out_folder = "./output_xml/"
    # filename = xml_out_folder + "DJI_0005-0019.xml"

    txt_folder = "./output_txt/"

    file_list = glob.glob(txt_folder + '*.txt')
    print("file_list", file_list)

    parser = Txt_To_Xml(class_name)

    for txt_path in file_list:
        parser.parse_txt(txt_folder, txt_path, xml_out_folder)


if __name__ == '__main__':
    main()