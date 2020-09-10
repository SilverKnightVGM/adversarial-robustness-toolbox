import os
import numpy as np
from PIL import Image
import re
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
import platform
import shutil


def move_to_folder(source_folder, dest_folder):
    
    # print("Source:", source_folder)
    # print("Destination:", dest_folder)
    
    os.makedirs(dest_folder, exist_ok=True)
    
    files = (file for file in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, file)))
    
    for file in files:
        shutil.move(os.path.join(source_folder, file), os.path.join(dest_folder, file))
    
    return

def load_dataset_temp(return_set, path, greebles_mode=2):
    '''
    path = folder containing the test and train subfolders for the dataset
    greebles_mode = default is 2. Basically how specific we want to be with categories. Mode 2 is just male/female and family type labeling.
    return_set = {train, test}
    --------
    Returns a tuple, containing two arrays one with the images and the another with the corresponding labels.
    '''
    
    if (platform.system() == "Windows"):
        path_train = path + "\\train"
        path_test = path + "\\test"
    else:
        path_train = path + "/train"
        path_test = path + "/test"

    train_filenames = os.listdir(path_train)
    if return_set == "test":
        test_filenames = os.listdir(path_test)

    # Remove alpha channel from png file, just keep the first 3 channels
    if (platform.system() == "Windows"):
        if return_set == "train":
            # train_images = np.array([np.array(Image.open(path_train + "\\" + fname))[...,:3] for fname in train_filenames])
            train_images = np.array([np.array(Image.open(path_train + "\\" + fname).convert('L')) for fname in train_filenames])
            train_images = np.expand_dims(train_images, axis=-1)
        elif return_set == "test":
            # eval_images = np.array([np.array(Image.open(path_test + "\\" + fname))[...,:3] for fname in test_filenames])
            eval_images = np.array([np.array(Image.open(path_test + "\\" + fname).convert('L')) for fname in test_filenames])
            eval_images = np.expand_dims(eval_images, axis=-1)
    else:
        if return_set == "train":
            train_images = np.array([np.array(Image.open(path_train + "/" + fname))[...,:3] for fname in train_filenames])
        elif return_set == "test":
            eval_images = np.array([np.array(Image.open(path_test + "/" + fname))[...,:3] for fname in test_filenames])

    '''
    File names denote the individual Greeble by defining the specific origin of the body type and parts, as well as its gender.

    The first character is the gender (m/f)

    The second number is the family (defined by body type, 1-5)
    Next there is a tilda (~) (is this referring to the dash in the filename?)

    The next few numbers describe where the parts came from in terms of the original Greebles.

    The third number is the family these particular parts ORIGINALLY came from. That is, a "2" would denote that the parts in the Greeble you are dealing with came from family 2 (1-5)

    The final number is which set of parts were taken from the specified family. Note that genders are never crossed (!), so that the number here only refers to the same gender parts as the Greeble you are dealing with. Depending on the number of individual Greebles in the original set, there could more more or less of these part sets (1-10, where 10 is the max possible as of August 2002).

    For example, "f1~16.max" is the model of a female Greeble of family 1, with body parts from family 1, set 6.
    '''

    if return_set == "train":
        train_labels = np.zeros(len(train_filenames), dtype='int32')
        train_labels_temp = np.zeros(len(train_filenames), dtype=object)
        for idx, fname in enumerate(train_filenames):
            l = np.empty(greebles_mode,dtype=object)
            s = "-"
            #replace all non alphanumeric characters with nothing
            label = re.sub('[^A-Za-z0-9]+', '', fname)
            #match label structure
            matchObj = re.match( r'(f|m)([1-5]{1})([1-5]{1})(10|[1-9])', label, re.M|re.I)
            if matchObj:
                #male of female
                if(greebles_mode >=1):
                    l[0] = matchObj.group(1)
                #body type, 1-5
                if(greebles_mode >=2):
                    l[1] = matchObj.group(2)
                #original family, 1-5
                if(greebles_mode >=3):
                    l[2] = matchObj.group(3)
                #which set of parts, 1-10
                if(greebles_mode >=4):
                    l[3] = matchObj.group(4)
            else:
               raise NameError('Wrong file name structurem check the greebles documentation.')
            s = s.join(l)
            train_labels_temp[idx] = s
    
    elif return_set == "test":
        
        eval_labels = np.zeros(len(test_filenames), dtype='int32')
        eval_labels_temp = np.zeros(len(test_filenames), dtype=object)
        for idx, fname in enumerate(test_filenames):
            l = np.empty(greebles_mode,dtype=object)
            s = "-"
            #replace all non alphanumeric characters with nothing
            label = re.sub('[^A-Za-z0-9]+', '', fname)
            #match label structure
            matchObj = re.match( r'(f|m)([1-5]{1})([1-5]{1})(10|[1-9])', label, re.M|re.I)
            if matchObj:
                #male of female
                if(greebles_mode >=1):
                    l[0] = matchObj.group(1)
                #body type, 1-5
                if(greebles_mode >=2):
                    l[1] = matchObj.group(2)
                #original family, 1-5
                if(greebles_mode >=3):
                    l[2] = matchObj.group(3)
                #which set of parts, 1-10
                if(greebles_mode >=4):
                    l[3] = matchObj.group(4)
            else:
               raise NameError('Wrong file name structurem check the greebles documentation.')
            s = s.join(l)
            eval_labels_temp[idx] = s

    filtered_all_labels = list(set(train_filenames)).sort()
    # lb = OneHotEncoder(categories=filtered_all_labels)
    lb = LabelBinarizer()
    
    if return_set == "train":
        train_labels = np.asarray(lb.fit_transform(train_labels_temp))
        train_data = (train_images, train_labels)
        return train_data
    elif return_set == "test":
        eval_labels = np.asarray(lb.fit_transform(eval_labels_temp))
        eval_data = (eval_images, eval_labels)
        return eval_data