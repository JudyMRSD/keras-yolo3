#! /usr/bin/env python
# python train.py -c ./aerial_zoo/config_aerial_4_class.json
import argparse
import os

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import json
from voc import parse_voc_annotation
from yolo3Label import LabelParser
from yolo import create_yolov3_model, dummy_loss
from generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs
from utils.train_monit import TrainMonitorTools
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from callbacks import CustomModelCheckpoint
from utils.multi_gpu_model import multi_gpu_model
import tensorflow as tf
import keras
import cv2
from keras.models import load_model


Annotation_Type = 'darknet_yolo3'
Classes = ['car', 'truch', 'bus', 'minibus']
Plot_Training_Instances = False

def plot_training_instances(train_ints, train_labels ):
    print("train_ints[0]['object'] ", train_ints[0]['object']) # train_ints[0]['object']  [{'name': 'car', 'xmin': 3183, 'ymin': 1337, 'xmax': 3292, 'ymax': 1408}, {'name': 'car', 'xmin': 2487, 'ymin': 2079, 'xmax': 2536, 'ymax': 2183}, {'name': 'car', 'xmin': 2394, 'ymin': 2103, 'xmax': 2451, 'ymax': 2244}, {'name': 'car', 'xmin': 2477, 'ymin': 1919, 'xmax': 2534, 'ymax': 2040}, {'name': 'car', 'xmin': 2400, 'ymin': 1929, 'xmax': 2450, 'ymax': 2044}, {'name': 'car', 'xmin': 2161, 'ymin': 1482, 'xmax': 2272, 'ymax': 1531}, {'name': 'car', 'xmin': 2745, 'ymin': 1286, 'xmax': 2869, 'ymax': 1352}, {'name': 'car', 'xmin': 2543, 'ymin': 1158, 'xmax': 2642, 'ymax': 1245}, {'name': 'car', 'xmin': 2447, 'ymin': 836, 'xmax': 2499, 'ymax': 937}, {'name': 'car', 'xmin': 2278, 'ymin': 893, 'xmax': 2331, 'ymax': 1008}, {'name': 'car', 'xmin': 2216, 'ymin': 749, 'xmax': 2263, 'ymax': 865}, {'name': 'car', 'xmin': 2281, 'ymin': 552, 'xmax': 2325, 'ymax': 658}]
    print("train_ints[0]['filename'] ", train_ints[0]['filename']) # train_ints[0]['filename']  ./training_data/aerial/images_may4/2300.jpg
    print("train_labels", train_labels) # {'car': 159, 'bus': 3}

    num_imgs = len(train_ints)
    for i in range(0, num_imgs):
        image_path = train_ints[i]['filename']
        print("img_path", image_path)
        image = cv2.imread(image_path)
        for box in train_ints[i]['object']:
            name = box['name']
            xmin = box['xmin']
            ymin = box['ymin']
            xmax = box['xmax']
            ymax = box['ymax']
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(image,
                        name,
                        (xmin, ymin - 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image.shape[0],
                        (0, 255, 0), 2)
        base = os.path.basename(image_path)
        out_path = './plot_training/'+base[:-4] + '_detected' + base[-4:]
        cv2.imwrite(out_path, (image).astype('uint8'))




def create_training_instances(
    train_annot_folder,
    train_image_folder,
    train_cache,
    valid_annot_folder,
    valid_image_folder,
    valid_cache,
    labels,
):
    # print("args: ", train_annot_folder, train_image_folder, train_cache, valid_annot_folder, valid_image_folder,valid_cache,labels)
    # parse annotations of the training set
    if Annotation_Type == 'voc':
        train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, train_cache, labels)
    elif Annotation_Type == 'darknet_yolo3':
        parser = LabelParser()
        train_ints, train_labels = parser.parse_yolo_annotation(Classes, train_annot_folder, train_image_folder, train_cache, labels)

    # train_ints[0].object   =  [{'name':'car','xmin':3183,'ymin':1337,'xmax':3292,'ymax':1408}, ...]
    # train_ints[0].filename =  './training_data/aerial/imgs_may4/2300.jpg'
    #print("train_ints, train_labels", train_ints, train_labels)
    # parse annotations of the validation set, if any, otherwise split the training set
    
    if Plot_Training_Instances==True:
        plot_training_instances(train_ints, train_labels)

    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8*len(train_ints))
        np.random.seed(0)
        np.random.shuffle(train_ints)
        np.random.seed()

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

        #print("valid_ints:", valid_ints)

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t\t'  + str(train_labels) + '\n')
        print('Given labels: \t\t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image

def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    makedirs(tensorboard_logs)
    
    early_stop = EarlyStopping(
        monitor     = 'loss', 
        min_delta   = 0.01, 
        patience    = 5, 
        mode        = 'min', 
        verbose     = 1
    )
    checkpoint = CustomModelCheckpoint(
        saved_weights_name,# + '{epoch:02d}.h5', 
        model_to_save   = model_to_save,
        monitor         = 'loss', 
        verbose         = 1, 
        save_best_only  = True, 
        mode            = 'min', 
        period          = 1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'min',
        epsilon  = 0.01,
        cooldown = 0,
        min_lr   = 0
    )
    tensorboard = TensorBoard(
        log_dir                = tensorboard_logs,
        write_graph            = True,
        write_images           = True,
    )    
    return [early_stop, checkpoint, reduce_on_plateau, tensorboard]

def create_model(
    nb_class, 
    anchors, 
    max_box_per_image, 
    max_grid, batch_size, 
    warmup_batches, 
    ignore_thresh, 
    multi_gpu, 
    saved_weights_name, 
    lr,
    scales
):
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            template_model, infer_model = create_yolov3_model(
                nb_class            = nb_class, 
                anchors             = anchors, 
                max_box_per_image   = max_box_per_image, 
                max_grid            = max_grid, 
                batch_size          = batch_size//multi_gpu, 
                warmup_batches      = warmup_batches,
                ignore_thresh       = ignore_thresh,
                scales              = scales
            )
    else:
        template_model, infer_model = create_yolov3_model(
            nb_class            = nb_class, 
            anchors             = anchors, 
            max_box_per_image   = max_box_per_image, 
            max_grid            = max_grid, 
            batch_size          = batch_size, 
            warmup_batches      = warmup_batches,
            ignore_thresh       = ignore_thresh,
            scales              = scales
        )

        


    # aerial_model = load_model('aerial_model.h5')
    # print("aerial_model summary", aerial_model.summary())
    # backend_model= load_model('backend.h5')
    # print("backend_model summary", backend_model.summary())

    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name): 
        # print("\nLoading pretrained weights.\n")
        # print("saved_weights_name", saved_weights_name)
        # template_model.load_weights(saved_weights_name)
        template_model.load_weights("backend.h5", by_name=True)
        print("summary",template_model.summary())
    else:

        template_model.load_weights("backend.h5", by_name=True)       

    if multi_gpu > 1:
        train_model = multi_gpu_model(template_model, gpus=multi_gpu)
    else:
        train_model = template_model      

    optimizer = Adam(lr=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)             

    return train_model, infer_model

def _main_(args):
    config_path = args.conf
    print("config_path", config_path)
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################
    train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
        config['train']['train_annot_folder'],
        config['train']['train_image_folder'],
        config['train']['cache_name'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['valid']['cache_name'],
        config['model']['labels']
    )
    print('\nTraining on the following labels: ' + str(labels))

    ###############################
    #   Create the generators
    ###############################    
    train_generator = BatchGenerator(
        instances           = train_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.3, 
        norm                = normalize
    )
    
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize
    )

    ###############################
    #   Create the model 
    ###############################
    if os.path.exists(config['train']['saved_weights_name']): 
        config['train']['warmup_epochs'] = 0
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times']*len(train_generator))   

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    multi_gpu = len(config['train']['gpus'].split(','))
    print("labels", labels)
    train_model, infer_model = create_model(
        nb_class            = len(labels),
        # nb_class            = 2,
        anchors             = config['model']['anchors'], 
        max_box_per_image   = max_box_per_image, 
        max_grid            = [config['model']['max_input_size'], config['model']['max_input_size']], 
        batch_size          = config['train']['batch_size'], 
        warmup_batches      = warmup_batches,
        ignore_thresh       = config['train']['ignore_thresh'],
        multi_gpu           = multi_gpu,
        saved_weights_name  = config['train']['saved_weights_name'],
        lr                  = config['train']['learning_rate'],
        scales              = config['train']['scales'],
    )

    ###############################
    #   Kick off the training
    ###############################
    callbacks = create_callbacks(config['train']['saved_weights_name'], config['train']['tensorboard_dir'], infer_model)

    history = train_model.fit_generator(
        generator        = train_generator, 
        steps_per_epoch  = len(train_generator) * config['train']['train_times'], 
        epochs           = config['train']['nb_epochs'] + config['train']['warmup_epochs'], 
        verbose          = 2 if config['train']['debug'] else 1,
        callbacks        = callbacks, 
        workers          = 4,
        max_queue_size   = 8
    )

    trainMonitTool = TrainMonitorTools()
    trainMonitTool.visualizeTrain(history)



    # make a GPU version of infer_model for evaluation
    if multi_gpu > 1:
        infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Run the evaluation
    ###############################   
    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))           

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')   

    args = argparser.parse_args()
    _main_(args)
