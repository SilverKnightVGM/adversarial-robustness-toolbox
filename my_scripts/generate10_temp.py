import os
import numpy as np
import platform
import subprocess
import shutil
import load10_temp as ltemp

render_util_folder = os.getcwd()
blender_folder = r"D:\Users\Enzo\Desktop\poke102938\School\RIT No Sync\Research\GAN\blender-2.78c-windows64" 

dataset_folder = r"D:\Users\Enzo\Desktop\poke102938\School\RIT No Sync\Research\GAN\greebles-generator-master\Greebles_3DS_10"
render_folder = os.path.join(os.getcwd(), "greebles_10") #reduced dataset of 10

if os.path.exists(render_folder) and os.path.isdir(render_folder):
    shutil.rmtree(render_folder)
    
if not os.path.isdir(dataset_folder):
    raise OSError(2, 'No such directory', str(dataset_folder))
if not os.path.exists(os.path.join(render_util_folder, 'render.py')):
    raise OSError(2, 'render.py not found', str(render_util_folder))

render_script = os.path.join(render_util_folder, 'render.py')

# Find the correct version of blender depending on the OS
if (platform.system() == "Windows"):
    blender_exec = os.path.join(blender_folder, 'blender.exe')
else:
    blender_exec = os.path.join(blender_folder, 'blender')
    
if not os.path.exists(os.path.join(blender_folder, blender_exec)):
    raise OSError(2, 'Blender executable not found', str(blender_folder))

# If you want to split command automatically
# import shlex; shlex.split("/bin/prog -i data.txt -o \"more data.txt\"")
# https://janakiev.com/blog/python-shell-commands/


set_mode = 'specific_all'
set_type = 'specific_angle'
#####################################
# Generate training set
blender_process = subprocess.run([blender_exec, '-b', '-P', render_script, '--', 
                    '-st', set_type, '-sm', set_mode, '-rp', render_folder, '-dp', dataset_folder, '-rm', 'none', '-pf', 'tensorflow',
                    '-is', '32', '-ni', '1', '-xr', '0', '-yr', '0', '-zr', '0'])

# Move training set to folder                    
source_folder = os.path.join(render_folder, "greebles_tf-" + set_mode, set_type)
ltemp.move_to_folder(source_folder, os.path.join(source_folder, "train"))

(x_train, y_train) = ltemp.load_dataset_temp("train", source_folder)
y_train = np.array(y_train, dtype='float32')
#####################################
# Generate test set
for z in range(-90,90+1,90):
    
    blender_process = subprocess.run([blender_exec, '-b', '-P', render_script, '--', 
                        '-st', set_type, '-sm', set_mode, '-rp', render_folder, '-dp', dataset_folder, '-rm', 'none', '-pf', 'tensorflow',
                        '-is', '32', '-ni', '1', '-xr', '0', '-yr', '0', '-zr', str(z)])

    # Move training set to folder                    
    source_folder = os.path.join(render_folder, "greebles_tf-" + set_mode, set_type)
    ltemp.move_to_folder(source_folder, os.path.join(source_folder, "test"))
    
    print(z)
    (x_test, y_test) = ltemp.load_dataset_temp("test", source_folder)
    y_test = np.array(y_test, dtype='float32')