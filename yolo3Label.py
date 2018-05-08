import os
import glob
import pickle
import cv2
import numpy as np



class LabelParser:
    def __init__(self):
        self.a = 1
    def parse_yolo_annotation(self,classes, ann_dir, img_dir, cache_name, labels=[]):
        self.classes = classes

        if os.path.exists(cache_name):
            print("exists", cache_name)
            with open(cache_name, 'rb') as handle:
                cache = pickle.load(handle)
            all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
        else:
            all_insts = []
            seen_labels = {}
            ann_list = sorted(glob.glob(ann_dir+'*.txt'))
            for ann in ann_list:
                print("ann", ann)
                img = {'object': []}

                ann_base = os.path.basename(ann)
                print("ann_base", ann_base) # DJI_0005-0018.txt
                image_path_jpg = img_dir + ann_base.replace('.txt','.jpg')
                image_path_JPG = img_dir + ann_base.replace('.txt','.JPG')
                image_path_png = img_dir + ann_base.replace('.txt','.png')
                if os.path.exists(image_path_jpg):
                    image_path = image_path_jpg
                elif os.path.exists(image_path_JPG):
                    image_path = image_path_JPG
                elif os.path.exists(image_path_png):
                    image_path = image_path_png
                else:
                    continue
                image = cv2.imread(image_path)
                img['filename'] = image_path
                self.img_height, self.img_width, img_channels = image.shape
                label_len = 5  # class_id, center_x, center_y, width, height


                # i4     # 32-bit signed integer
                # 'f8'   # 64-bit floating-point number

                # 'class_id', 'center_x', 'center_y','bbox_width','bbox_height'
                bbox_labels = np.loadtxt(ann, delimiter=' ',
                                              dtype = {'names': ('class_id', 'center_x', 'center_y','bbox_width','bbox_height'),
                                                       'formats': ('i4', 'f4', 'f4','f4','f4')})

                for label in bbox_labels:
                    label_object = self.yolo_to_vertex(label)
                    if label_object['name'] in seen_labels:
                        seen_labels[label_object['name']] += 1
                    else:
                        seen_labels[label_object['name']] = 1

                    img['object'].append(label_object)

                if len(img['object']) > 0:
                    all_insts += [img]

                # dict
                # img = {'object':[], 'filename':'./training_data/aerial/images_may4/2300.jpg', 'width':4156, 'height':2793}

                #'object': [{'name':'car', 'xmin':3183, 'ymin':1337, 'xmax':3292}]
            print(" finish ")
            cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
            with open(cache_name, 'wb') as handle:
                pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return all_insts, seen_labels



    def yolo_to_vertex(self, label):
        label_object = {}
        label_object['name'] = self.classes[label[0]]
        # 'class_id', 'center_x', 'center_y','bbox_width','bbox_height'
        label_object['xmin'] = int((label[1] - label[3] / 2) * self.img_width)
        label_object['ymin'] = int((label[2] - label[4] / 2) * self.img_height)
        label_object['xmax'] = int((label[1] + label[3] / 2) * self.img_width)
        label_object['ymax'] = int((label[2] + label[4] / 2) * self.img_height)
        print("label_object", label_object)
        return label_object

def main():
    # classes = ['car', 'truch', 'bus', 'minibus']
    classes = ['car', 'car', 'car', 'car']
    ann_dir = './training_data/yolo3_darknet_aerial/'
    img_dir = './training_data/yolo3_darknet_aerial/'
    cache_name = './training_data/yolo3_darknet_aerial.pkl'
    parser = LabelParser()
    parser.parse_yolo_annotation(classes, ann_dir, img_dir, cache_name, labels=[])


if __name__ == '__main__':
    main()







