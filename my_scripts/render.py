# Generate Greeble dataset
#
# Run as TO USE ARGPARSE: blender -b -P render.py -- -arg1 test
# This is the same as = blender --background --python render.py

# "blender-2.78c-windows64\blender.exe" -b -P render.py -- -st test -sm original -rm slight -pf keras -is 32 -ni 8000

# If no arguments are passed the default values below apply

import bpy
import os
import sys
import mathutils
import random
from math import radians
import argparse
import json

# adjustable parameters
r = 15  # distance of camera to greeble
set_type = "train" # options: "train", "test"
set_mode = "flat_range_90" #options: original, flat_range, new_flat, upside_down
light_mode = 'behind_camera' #options: original, behind_camera, at_camera
rotation_mode = 'random' #options: random, slight, none
path_format = 'tensorflow' #options: tensorflow, keras
# imsize = 96  # size of output image
imsize = 32  # size of output image
num_imgs = 8000

random.seed(1337)
        
# original windows paths
orig_path = os.getcwd() + "/Greebles-2-0-symmetric/Greebles 3DS"
if path_format == 'tensorflow':
    render_path = os.getcwd() + "/greebles_tf" + "-" + set_mode + "/" + set_type + "/"
else:
    render_path = os.getcwd() + "/greebles_keras" + "-" + set_mode + "/" + set_type + "/"

NUM_GREEBLES = 80
POSES_PER_GREEBLE = num_imgs // 80
ORIGIN = (0, 0, 0)

# Mapping of body parts by their names inside the original files
parts_mapping = {
    'Family_1':{
        "F1":{
            "BODY":["REVOLVE-01.001"],
            "NOSE":["REVOLVE-01.002", "REVOLVE-01.008", "REVOLVE-01.007", "REVOLVE-01.006", "REVOLVE-01.005", "REVOLVE-01.004", "REVOLVE-01.003"],
            "RIGHT_EAR":["REVOLVE-01.009"],
            "LEFT_EAR":["REVOLVE-01"],
            "BUMP":["REVOLVE-01.010"]
        },
        "M1":{
            "BODY":["REVOLVE-01.001"],
            "NOSE":["REVOLVE-01.005", "REVOLVE-01", "REVOLVE-01.009", "REVOLVE-01.008", "REVOLVE-01.007", "REVOLVE-01.006"],
            "RIGHT_EAR":["REVOLVE-01.002"],
            "LEFT_EAR":["REVOLVE-01.003"],
            "BUMP":["REVOLVE-01.004"]
        }
    },
    'Family_2':{
        "F1":{
            "BODY":["REVOLVE-01.001"],
            "NOSE":["REVOLVE-01"],
            "RIGHT_EAR":["REVOLVE-01.002"],
            "LEFT_EAR":["REVOLVE-01.003"],
            "BUMP":["REVOLVE-01.004"]
        },
        "M1":{
            "BODY":["REVOLVE-01.001"],
            "NOSE":["REVOLVE-01"],
            "RIGHT_EAR":["REVOLVE-01.002"],
            "LEFT_EAR":["REVOLVE-01.003"],
            "BUMP":["REVOLVE-01.004"]
        }
    },
    'Family_3':{
        "F1":{
            "BODY":["REVOLVE-01.001"],
            "NOSE":["REVOLVE-01.002"],
            "RIGHT_EAR":["REVOLVE-01"],
            "LEFT_EAR":["REVOLVE-01.004"],
            "BUMP":["REVOLVE-01.003"]
        },
        "M1":{
            "BODY":["REVOLVE-01.001"],
            "NOSE":["REVOLVE-01"],
            "RIGHT_EAR":["REVOLVE-01.002"],
            "LEFT_EAR":["REVOLVE-01.003"],
            "BUMP":["REVOLVE-01.004"]
        }
    },
    'Family_4':{
        "F1":{
            "BODY":["body type .001"],
            "NOSE":["body type "],
            "RIGHT_EAR":["body type .002"],
            "LEFT_EAR":["body type .003"],
            "BUMP":["body type .004"]
        },
        "M1":{
            "BODY":["body type .001"],
            "NOSE":["body type "],
            "RIGHT_EAR":["body type .002"],
            "LEFT_EAR":["body type .003"],
            "BUMP":["body type .004"]
        }
    },
    'Family_5':{
        "F1":{
            "BODY":["body type .001", "body type "],
            "NOSE":["body type .005"],
            "RIGHT_EAR":["body type .002"],
            "LEFT_EAR":["body type .003"],
            "BUMP":["body type .004"]
        },
        "M1":{
            "BODY":["body type .001", "body type "],
            "NOSE":["body type .005"],
            "RIGHT_EAR":["body type .002"],
            "LEFT_EAR":["body type .003"],
            "BUMP":["body type .004"]
        }
    }
}

# https://docs.blender.org/api/blender_python_api_2_64_4/bpy.types.Material.html
def makeMaterial(name, diffuse, specular, alpha):
    mat=bpy.data.materials.new(name)
    mat.diffuse_color = diffuse
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 0.9
    mat.specular_color = specular
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.0
    mat.specular_hardness = 1
    mat.alpha = alpha
    mat.ambient = 1
    return mat

def modifyMaterial(mat):
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 0.98
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.0
    mat.specular_hardness = 1
    return mat
    

def setMaterial(ob, mat):
    me = ob.data
    me.materials.append(mat)

def random_angle(min_angle=0, max_angle=360, step=1):
    return radians((max_angle - min_angle) * random.random() + min_angle)

def flatten(lst):
    for item in lst:
        if isinstance(item, list) and not isinstance(item, str):
            yield from flatten(item)
        else:
            yield item

def flatten_list(lst):
    return list(flatten(lst))

def delete_obj(label):
    obj = bpy.data.objects.get(label)
    if obj is not None:
        obj.select = True
        bpy.ops.object.delete()

def split_parts(greeble, filename, parts_to_remove=['NOSE', 'BUMP']):
    """Splits the greeble model into parts in order to remove some of them

    Args:
        greeble ([bpy_types.Object]): greeble blender object
        filename ([str]): filename (not path) of greeble with extension
        parts_to_remove (list, optional): List of parts to remove, Possible values are ['BUMP', 'NOSE', 'LEFT_EAR', 'BODY', 'RIGHT_EAR']. Defaults to ['NOSE', 'BUMP'].
    """
    gender = filename[0].upper()
    family = filename[1]
    parts_to_remove = [item.upper() for item in parts_to_remove]

    # import pdb; pdb.set_trace()

    parts = greeble
    # print("PARTS TYPE:", parts.type)
    bpy.ops.mesh.separate(type='LOOSE')
    parts = bpy.context.selected_objects
    # sort by number of verts (last has most)
    parts.sort(key=lambda o: len(o.data.vertices))
      
    part_names = []
    for part in parts:
        print(part.name, len(part.data.vertices))
        part_names.append(part.name)
        
    # Get the actual part names with the mapping
    parts_to_remove_map = []
    for part in parts_to_remove:
        n = parts_mapping['Family_'+family][gender+'1'][part]
        parts_to_remove_map.append(n)
    
    parts_to_remove_map = flatten_list(parts_to_remove_map)
    
    parts_to_pop = []
    for idx, pn in enumerate(part_names):
        if pn in parts_to_remove_map:
            continue
        else:
            # parts to keep later
            parts_to_pop.append(idx)

    # pop off part to be kept
    for i in reversed(parts_to_pop):
        parts.pop(i)

    # remove the rest
    for o in parts:
        bpy.data.objects.remove(o, do_unlink=True)

def add_lamp(lamp_name, lamp_type, radius=r, mode=light_mode):
    # adapted from Stack Overflow:
    # https://stackoverflow.com/questions/17355617/can-you-add-a-light-source-in-blender-using-python
    
    # data = bpy.data.lamps.new(name=lamp_name, type=lamp_type)
    # Hemi light is the most even
    data = bpy.data.lamps.new(name=lamp_name, type="HEMI")
    
    if mode == 'original':
        pass
    elif mode == 'behind_camera':
        data.energy = 10
    else:
        raise NotImplementedError("Unexpected lamp mode")
    
    lamp_object = bpy.data.objects.new(name=lamp_name, object_data=data)
    bpy.context.scene.objects.link(lamp_object)
    
    if mode == 'original':
        lamp_object.location = (0, 0, radius)
    elif mode == 'behind_camera':
        lamp_object.location = (0, -(radius+1), 0)
    elif mode == 'at_camera':
        lamp_object.location = (0, 0, radius)
    else:
        raise NotImplementedError("Unexpected lamp mode")
    
    return lamp_object


def point_to_origin(obj):
    direction = -mathutils.Vector(obj.location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()


def render(greeble, f, lamp_type, lamp_empty=None):
    for i in range(POSES_PER_GREEBLE):
        # rotate greeble randomly
        # When rotating on an axis, the other two remain constant
        x_angle = random_angle(min_angle=-30, max_angle=30)
        y_angle = random_angle(min_angle=-30, max_angle=30)
        
        if set_mode == "specific" or set_mode == "specific_all":
            if set_type == "specific_angle":
                greeble.rotation_euler = (radians(xr), radians(yr), radians(zr))
            else:
                raise AttributeError("You need to specify the XYZ angles of rotation")
        elif set_mode == "original":
            if set_type == "train":
                greeble.rotation_euler = (0, 0, random_angle())
            if set_type == "test":
                greeble.rotation_euler = (x_angle, y_angle, random_angle())
        elif set_mode == "flat_range":
            if set_type == "train":
                greeble.rotation_euler = (0, 0, random_angle(min_angle=0, max_angle=0))
            if set_type == "test":
                greeble.rotation_euler = (0, 0, random_angle(min_angle=220, max_angle=260))
        elif set_mode == "new_flat":
            if set_type == "train":
                greeble.rotation_euler = (0, 0, random_angle(min_angle=-30, max_angle=30))
            if set_type == "test":
                greeble.rotation_euler = (0, 0, random_angle(min_angle=-60, max_angle=60))
        elif set_mode == "upside_down":
            if set_type == "train":
                greeble.rotation_euler = (radians(180), 0, 0)
            if set_type == "test":
                greeble.rotation_euler = (0, 0, random_angle(min_angle=220, max_angle=260))
        elif set_mode == "flat_range_90":
            if set_type == "train":
                greeble.rotation_euler = (0, 0, 0)
            if set_type == "test":
                greeble.rotation_euler = (0, 0, radians(90))
                   
        # rotate lamp randomly
        if lamp_empty is not None:
            if rotation_mode == 'random':
                mat_rot = mathutils.Euler((random_angle(), random_angle(), random_angle()), 'XYZ')
                mat_rot = mat_rot.to_matrix().to_4x4()
                lamp_empty.matrix_world = mat_rot
            elif rotation_mode == 'slight':
                x_ang = 10
                y_ang = 50
                z_ang = -50
                mat_rot = mathutils.Euler((random_angle(x_ang, x_ang), random_angle(y_ang, y_ang), random_angle(z_ang, z_ang)), 'XYZ')
                mat_rot = mat_rot.to_matrix().to_4x4()
                lamp_empty.matrix_world = mat_rot
            else:
                pass

        # Split and remove parts
        if rmparts is not None:
            split_parts(greeble, f, parts_to_remove=rmparts)
            parts_str = "-".join(rmparts)
            lamp_type =  lamp_type + "_" + parts_str
        
        if path_format == 'tensorflow':
            bpy.context.scene.render.filepath = "{}{}_{}_{:03d}.png".format(render_path, f[:-4], lamp_type, i)
        else:
            bpy.context.scene.render.filepath = "{}{}_{}_{:03d}.png".format(render_path+f[:2]+"/", f[:-4], lamp_type, i)
        
        bpy.ops.render.render(write_still=True)
        
    return greeble


def process_greeble(greeble, root, f):
    delete_obj("Cube")  # delete default cube
    delete_obj("Lamp")  # delete default lamp

    # delete previous greeble
    if greeble is not None:
        greeble.select = True
        bpy.ops.object.delete()

    # import .3ds file
    fpath = os.path.join(root, f)
    bpy.ops.import_scene.autodesk_3ds(filepath=fpath)

    # recenter
    # on import, all previously selected objects are deselected and the newly imported object is selected
    greeble = bpy.context.selected_objects[0]
    new_origin = (0, 0, greeble.dimensions[2] / 2)  # place center at median of greeble height (z-dimension)
    bpy.context.scene.cursor_location = new_origin
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    greeble.location = ORIGIN
    
    # Material
    # mat_color = makeMaterial('m_color', (0,0,0), (0,0,0), 0)
    mod_mat = modifyMaterial(greeble.active_material)

    greeble.active_material = mod_mat
    
    # No shadows
    # greeble.active_material.use_shadeless = True

    # set camera location
    camera = bpy.data.objects["Camera"]
    camera.location = (0, -r, 0)

    # point camera to origin
    point_to_origin(camera)

    # create empty (for lamp orbit)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = ORIGIN

    # top lamp
    # top_lamp = add_lamp("Top_Lamp", 'SPOT')

    # render for lamp configs
    random_lamp = add_lamp("Random_Lamp", 'POINT')
    # random_lamp = add_lamp("Random_Lamp", 'SPOT')
    random_lamp.parent = b_empty
    render(greeble, f, "lamps", lamp_empty=b_empty)

    delete_obj("Random_Lamp")
    delete_obj("Top_Lamp")

    return greeble


# get the args passed to blender after "--", all of which are ignored by
# blender so scripts may receive their own arguments
argv = sys.argv

# https://developer.blender.org/diffusion/B/browse/master/release/scripts/templates_py/background_job.py
if "--" not in argv:
    argv = []  # as if no args are passed
else:
    argv = argv[argv.index("--") + 1:]  # get all args after "--"

    # Create the parser
    parser = argparse.ArgumentParser(description='Render 3D models to images. When rotating on an axis, the other two remain constant.')

    # Add the arguments
    parser.add_argument(
        "-r", "--camera_distance", dest="r", type=int, required=False, default=15,
        help="Distance of camera to greeble. Default is 15."
    )

    parser.add_argument(
        "-st", "--set_type", dest="set_type", type=str, required=True,
        choices=['specific_angle', 'train', 'test'],
        help="Generate 'specific' sample(s), 'train' or 'test' set. If 'specific_angle' REMEMBER to set the parameter -ni/--num_imgs to how many total samples you want to generate. These are generated from a random selection of greebles."
    )

    parser.add_argument(
        "-sm", "--set_mode", dest="set_mode", type=str, required=True,
        choices=['original', 'flat_range', 'new_flat', 'upside_down', 'specific', 'specific_all'],
        help="Mode to use. With 'specific' you can get individual samples."
    )

    parser.add_argument(
        "-lm", "--light_mode", dest="light_mode", type=str, required=False, default='behind_camera', choices=['original', 'behind_camera', 'at_camera'],
        help="Light mode to use."
    )

    parser.add_argument(
        "-rm", "--rotation_mode", dest="rotation_mode", type=str, required=False, default='none', choices=['random', 'slight', 'none'],
        help="Rotation mode to use for the lamp."
    )

    parser.add_argument(
        "-pf", "--path_format", dest="path_format", type=str, required=False, default='keras', choices=['keras', 'tensorflow'],
        help="Folder structure. Keras puts each family in a separate folder."
    )

    parser.add_argument(
        "-is", "--imsize", dest="imsize", type=int, required=False, default=32,
        help="Size of output image, in pixels."
    )

    parser.add_argument(
        "-ni", "--num_imgs", dest="num_imgs", type=int, required=False, default=8000,
        help="Number of total images to produce across all models used."
    )

    parser.add_argument(
        "-dp", "--dataset_path", dest="dataset_path", type=str, required=True,
        help="Path to where the 3D models are located. The structure is: folder-> family_subfolders->3ds_files. Don't use trailing slash at the end of the path"
    )

    parser.add_argument(
        "-rp", "--render_path", dest="render_path", type=str, required=True,
        help="Path to put the rendered images in. Subfolders will be created inside here depending on the mode."
    )

    parser.add_argument(
        "-xr", "--xr_angle", dest="xr", type=int, required=False,
        help="X Angle to use for rotation."
    )

    parser.add_argument(
        "-yr", "--yr_angle", dest="yr", type=int, required=False,
        help="Y Angle to use for rotation."
    )

    parser.add_argument(
        "-zr", "--zr_angle", dest="zr", type=int, required=False,
        help="Z Angle to use for rotation."
    )

    parser.add_argument(
        "-p", "--parts_remove", nargs="*", dest="rmparts", type=str, required=False,
        choices=['BODY', 'NOSE', 'RIGHT_EAR', 'LEFT_EAR', 'BUMP'],
        help="Parts to remove from the greeble. Can input multiple, separate with spaces"
    )

    # Parse argv contents
    args = parser.parse_args(argv)

    # Recover values
    r = args.r
    set_type = args.set_type
    set_mode = args.set_mode
    light_mode = args.light_mode
    rotation_mode = args.rotation_mode
    path_format = args.path_format
    imsize = args.imsize
    num_imgs = args.num_imgs
    dataset_path = args.dataset_path
    render_path = args.render_path
    rmparts = args.rmparts
    # print(rmparts)
    # import pdb; pdb.set_trace()
    # Angles
    xr = args.xr
    yr = args.yr
    zr = args.zr
    
    # paths
    # orig_path = os.path.join(dataset_path, "Greebles 3DS")
    orig_path = os.path.normpath(dataset_path)
    orig_path = orig_path.rstrip("\\'\"") #strip any quotes or backslashes

    if path_format == 'tensorflow':
        path_format_name = "greebles_tf-"
    else:
        path_format_name = "greebles_keras-"
    
    render_path = os.path.join(render_path, path_format_name + set_mode, set_type, "")

# print(r)
# print(set_type)
# print(set_mode)
# print(light_mode)
# print(rotation_mode)
# print(path_format)
# print(imsize)
# print(num_imgs)

# create a scene
scene = bpy.data.scenes.new("Scene")

# rendered images should be square
bpy.context.scene.render.resolution_x = imsize * 2  # not sure why we have to double imsize
bpy.context.scene.render.resolution_y = bpy.context.scene.render.resolution_x

#https://docs.blender.org/api/2.78/bpy.types.RenderSettings.html
bpy.context.scene.render.alpha_mode = 'SKY'
# bpy.context.scene.render.alpha_mode = 'TRANSPARENT'

#https://blender.stackexchange.com/questions/32758/how-to-set-a-background-using-the-cycles-render-engine-with-the-api
# bpy.context.scene.render.engine = 'CYCLES'
# bpy.context.scene.world.use_nodes = True

# https://blender.stackexchange.com/questions/149710/python-render-object-with-transparent-background
# transparent background
# bpy.context.scene.render.film_transparent = True
# bpy.context.scene.render.image_settings.color_mode = 'RGBA'

# Dictionary to put all generated greeble details
master_render_dict = {}

# greeble object
curr_greeble = None

# complete for every .3ds file
all_filtered = []
for root, dirs, files in os.walk(orig_path):
    # pick out 3DS files
    current_filtered = list(filter(lambda x: x[-4:].lower() == ".3ds", files))
    filtered = []
    for f in current_filtered:
        filtered = os.path.join(root,f)
        all_filtered.append(filtered)

print(all_filtered)
print(len(all_filtered))

# input("CONTINUE???")

def random_choices(population, k=1):
    samples = []
    for i in range(k):
        samples.append(random.choice(population))
    return samples


if set_mode == 'specific': #pick a subset of greebles
    # Generate just one pose per choice of greeble
    POSES_PER_GREEBLE = 1
    filtered = random_choices(all_filtered, k=num_imgs)
elif set_mode == 'specific_all':
    # Generate just one pose per choice of greeble
    POSES_PER_GREEBLE = 1
    filtered = all_filtered

# filtered_dict = {"filename": curr_greeble}

# with open('test_list_filtered.txt', 'w') as filehandle:
    # filehandle.writelines("%s\n" % place for place in filtered)

for f in filtered:
    head, tail = os.path.split(f)
    curr_greeble = process_greeble(curr_greeble, head, tail)
    print(f)
    # input("---------")


# Write to file parameters to communicate back
# render_dict = {
                # "render_path" : render_path,
                # "orig_path": orig_path
                # "filename": curr_greeble
              # }

# with open('render_dict.json', 'w', encoding='utf-8') as f:
    # json.dump(master_render_dict, f, ensure_ascii=False, indent=4)