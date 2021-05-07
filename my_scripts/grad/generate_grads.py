# %%
import time
start_time = time.time()
# Hide warnings on Jupyter Notebook
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
# Use CPU only for Keras
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import platform

#this is what makes the plots transparent in notebooks
import matplotlib.pyplot as plt

from keras.models import load_model

from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam
from vis.visualization import visualize_cam_with_losses
from vis.visualization import visualize_activation
from vis.input_modifiers import Jitter
from vis.visualization import get_num_filters

from vis.utils import utils
from keras import activations

from sklearn.externals import joblib

import time

# %%
img_size = 224
# extra = 'rgb_none'
# extra = 'rgb_imagenet'

# extra = 'rgb_imagenet_noAT'
# extra = 'rgb_imagenet_AT'

# extra = 'rgb_imagenet_noAT2'
extra = 'rgb_imagenet_AT2'


if (platform.system() == "Windows"):
    model = load_model(os.path.join(os.getcwd(), '..', 'model.h5'))
else:
    # model = load_model(os.path.join(os.getcwd(), 'model.h5'))
    model = load_model(os.path.join(os.getcwd(), '..', 'model_{}_{}.h5'.format(img_size, extra)))
model.summary()


# %%
# print([layer.name for layer in model.get_layer('vgg16').layers])
# for layer in model.layers:
#     print(layer.name)
all_layer_names = [layer.name for layer in model.layers]
print(all_layer_names)


# %%
if (platform.system() == "Windows"):
    render_folder = os.path.join(os.getcwd(), "..","greebles_10") #reduced dataset of 10
else:
    # render_folder = os.path.join(os.getcwd(),"greebles_10") #reduced dataset of 10
    render_folder = os.path.join(os.getcwd(), "..", "greebles_10") 

set_mode = 'specific_all'
set_type = 'specific_angle'
source_folder = os.path.join(render_folder, "greebles_tf-" + set_mode, set_type)

# test_set_name = "greebles10_fgsm03_-90_90_1_224.npy".format(img_size)
test_set_name = "greebles10_test_-90_90_1_224.npy".format(img_size)
# test_set_name = "greebles10_upsidedown_-90_90_1_{}.npy".format(img_size)

labels = {0: 'f-1', 1: 'f-2', 2: 'f-3', 3: 'f-4', 4: 'f-5', 5: 'm-1', 6: 'm-2', 7: 'm-3', 8: 'm-4', 9: 'm-5'}
test_set_path = os.path.join(source_folder, test_set_name)
# all_test = np.load(test_set_path, allow_pickle=True).item()
all_test = joblib.load(test_set_path)


# %%
def to_rgb(img):
    '''https://github.com/keras-team/keras/issues/11208'''
    # img = img * 255
    # img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
    
    # img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.float32)
    img_rgb = np.asarray(np.dstack((img, img, img)))
    return img_rgb

def set_to_rgb(t_set):
    rgb_list = []
    #convert x_train data to rgb values
    for i in range(len(t_set)):
        rgb = to_rgb(t_set[i])
        rgb_list.append(rgb)
        #print(rgb.shape)
        rgb_arr = np.stack([rgb_list],axis=4)
    rgb_arr_to_3d = np.squeeze(rgb_arr, axis=4)
    return rgb_arr_to_3d

def get_test_angle_normal(all_test, angle, fgsm=False):
    (x_test, y_test) = np.copy(all_test[str(angle)][0]), np.copy(all_test[str(angle)][1])
    # If we are using the saved fgsm images, those are already normalized
    if fgsm == False:
        print("Dividing by 255...")
        x_test *= (1.0/255)
    x_test = set_to_rgb(x_test)
    return (x_test, y_test)


# %%
# Display one specific angle just as an example of what images are we using
angle_used = '10'
(x_test, y_test) = get_test_angle_normal(all_test, angle=angle_used)

# Debug
print(np.amin(x_test))
print(np.amax(x_test))

# Make background the same color
# x_test[x_test < 70] = 0
# x_test *= (1.0/255)
print(x_test.shape)

fig, axes = plt.subplots(2, 5, figsize=(15,6))
for i,ax in enumerate(axes.flat):
    ax.imshow(x_test[i], cmap='Greys_r')
    true_label = np.argwhere(y_test[i] == 1).flatten()
    ax.set_title("{number}: {label}".format(label=labels[int(true_label)], number=i), fontsize=15, color='#ba5e27')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(axis=u'both', which=u'both', length=0)

fig.suptitle('Testing Samples', fontsize=15, color='orange')

plt.show()


# %%
(x_test, y_test) = get_test_angle_normal(all_test, angle=0)
# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
# layer_idx = utils.find_layer_idx(model, 'dense_2')
layer_idx = utils.find_layer_idx(model, all_layer_names[-1])

# Example saliency visualization
class_idx = 0
indices = np.where(y_test[:, class_idx] == 1.)[0]
idx = indices[0] # pick some random input from here.

# Remove eventually to speed up
model.layers[layer_idx].activation = activations.softmax
model = utils.apply_modifications(model)

fig, ax = plt.subplots(1, 4, figsize=(15,6))
# ax[0].set_title(labels[class_idx], fontsize=15, color='#ba5e27')
ax[0].set_title(labels[class_idx], fontsize=15)
ax[0].imshow(x_test[idx], cmap='Greys_r')
ax[0].set_xticks([])
ax[0].set_yticks([])

pen_layer_name = 'block5_conv3'
penultimate_layer = utils.find_layer_idx(model, pen_layer_name)

for i, modifier in enumerate([None, 'guided', 'relu']):
    grads = visualize_saliency(model, layer_idx, filter_indices=class_idx,
                               seed_input=x_test[idx], backprop_modifier=modifier)
    # grads = visualize_cam(model, layer_idx, filter_indices=class_idx, 
    #                                         seed_input=x_test[idx], backprop_modifier=modifier,
    #                                         penultimate_layer_idx=penultimate_layer)
    if modifier == None:
        modifier = 'vanilla'
    # ax[i+1].set_title(modifier, color='#ba5e27', fontsize=15)
    ax[i+1].set_title(modifier, fontsize=15)
    ax[i+1].imshow(grads, cmap='jet')
    # ax[i+1].imshow(grads, cmap='Greys_r')
    ax[i+1].imshow(x_test[idx], cmap='Greys_r', alpha=0.10)
    ax[i+1].set_xticks([])
    ax[i+1].set_yticks([])
# fig.suptitle('Softmax', fontsize=15, color='orange', fontweight ="bold", y=0.8)
fig.suptitle('Softmax', fontsize=15, fontweight ="bold", y=0.8)
#############################################
# Swap softmax with linear for better results
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

fig, ax = plt.subplots(1, 4, figsize=(15,6))
# ax[0].set_title(labels[class_idx], fontsize=15, color='#ba5e27')
ax[0].set_title(labels[class_idx], fontsize=15)
ax[0].imshow(x_test[idx], cmap='Greys_r')
ax[0].set_xticks([])
ax[0].set_yticks([])

for i, modifier in enumerate([None, 'guided', 'relu']):
    grads = visualize_saliency(model, layer_idx, filter_indices=class_idx,
                               seed_input=x_test[idx], backprop_modifier=modifier)
    # grads = visualize_cam(model, layer_idx, filter_indices=class_idx, 
    #                                         seed_input=x_test[idx], backprop_modifier=modifier,
    #                                         penultimate_layer_idx=penultimate_layer)
    if modifier == None:
        modifier = 'vanilla'
    # ax[i+1].set_title(modifier, color='#ba5e27', fontsize=15)
    ax[i+1].set_title(modifier, fontsize=15)
    ax[i+1].imshow(grads, cmap='jet')
    # ax[i+1].imshow(grads, cmap='Greys_r')
    ax[i+1].imshow(x_test[idx], cmap='Greys_r', alpha=0.10)
    ax[i+1].set_xticks([])
    ax[i+1].set_yticks([])
# fig.suptitle('Linear', fontsize=15, color='orange', fontweight ="bold", y=0.8)
fig.suptitle('Linear', fontsize=15, fontweight ="bold", y=0.8)

plt.show()


# %%
# Change these as needed
class_idx = 1
indices = np.where(y_test[:, class_idx] == 1.)[0]
idx = indices[0]

method_vis = ['saliency', 'gradcam']
# method_vis = [s + "_fgsm" for s in method_vis]

#### SELECT HERE #####
method_vis_select = method_vis[1]
view_pred_class = [0, 1, 2, 'None'][0]
######################

save_filename = "grad_data"
os.makedirs(save_filename, exist_ok=True)
save_filename = os.path.join(save_filename, "{}{}_{}_{}_{}.npy".format(method_vis_select, view_pred_class, img_size, extra, test_set_name.split('_')[1]))

last_layer_name = all_layer_names[-1]
pen_layer_name = 'block5_conv3'

penultimate_layer = utils.find_layer_idx(model, pen_layer_name)
layer_idx = utils.find_layer_idx(model, last_layer_name)


# %%
print(save_filename)

all_gradcam = {}
print_flag = 1
for c, current_angle in enumerate(all_test.keys()):
    # if os.path.exists("{}/{}_sal_{}.png".format(sal_dir, str(c).zfill(3), current_angle)):
    #     print("Skipped {}".format(current_angle))
    #     continue
    
    # if (int(current_angle)%10) == 0: pass
    if True:
        if (int(current_angle)%10) == 0: print(current_angle, end=", ")
        if "fgsm" not in test_set_name.split('_')[1]:
            if print_flag:
                print("Normalizing...")
                print_flag = 0
            (x_test, y_test) = get_test_angle_normal(all_test, angle=current_angle, fgsm=False)
        else:
            (x_test, y_test) = get_test_angle_normal(all_test, angle=current_angle, fgsm=True)
        
        predictions = model.predict(x_test)
        y_pred = np.argmax(predictions, axis=1)
        
        class_gradcam = {}
        for class_idx in range(10):
            indices = np.where(y_test[:, class_idx] == 1.)[0]
            idx = indices[0]

            if method_vis[0] == method_vis_select:
                modifier = None
                if view_pred_class == 1:
                    grads = visualize_saliency(model, layer_idx, filter_indices=y_pred[idx], 
                                        seed_input=x_test[idx], backprop_modifier=modifier)
                else:
                    grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, 
                                        seed_input=x_test[idx], backprop_modifier=modifier)
                pen_layer_name = 'input'
            elif method_vis[1] == method_vis_select:
                modifier = None
                if view_pred_class == 1:
                    grads = visualize_cam(model, layer_idx, filter_indices=y_pred[idx],
                                            seed_input=x_test[idx], backprop_modifier=modifier,
                                            penultimate_layer_idx=penultimate_layer)
                else:
                    grads = visualize_cam(model, layer_idx, filter_indices=class_idx, 
                                            seed_input=x_test[idx], backprop_modifier=modifier,
                                            penultimate_layer_idx=penultimate_layer)
            class_gradcam[class_idx] = grads
        all_gradcam[current_angle] = class_gradcam


joblib.dump(all_gradcam, save_filename)

print("Done\n")


# %%
from datetime import datetime
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("now =", dt_string)
print("Elapsed time: {}".format(time.time() - start_time))
