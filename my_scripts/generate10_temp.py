import os
import platform
import subprocess

render_util_folder = os.getcwd()
blender_folder = r"D:\Users\Enzo\Desktop\poke102938\School\RIT No Sync\Research\GAN\blender-2.78c-windows64" 

dataset_folder = r"D:\Users\Enzo\Desktop\poke102938\School\RIT No Sync\Research\GAN\greebles-generator-master\Greebles_3DS_10"
render_folder = os.path.join(os.getcwd(), "greebles_10")
    
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
blender_process = subprocess.run([blender_exec, '-b', '-P', render_script, '--', 
                    '-st', 'specific_angle', '-sm', 'specific_all', '-rp', render_folder, '-dp', dataset_folder, '-rm', 'none', '-pf', 'keras',
                    '-is', '32', '-ni', '1', '-xr', '0', '-yr', '0', '-zr', '0'])