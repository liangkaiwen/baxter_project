#!/bin/python

import sys
import os
import os.path
import subprocess

this_filename_for_usage = 'luis_batch_folder.py'

# windows:
cwd = None
exe_file = 'C:/devlibs/object_modeling/batch_ready/volume_modeler_main'
# linux:
#cwd = '.'
#exe_file = os.path.join(cwd, 'volume_modeler_main')

command_string = exe_file
command_string += ' --use_f200_camera --turntable_rotation_limit 400  --circle_mask_pixel_radius 300  --frame_increment 1 --f --vs 0.001    --grid_size 64 --max 0.6  --mt grid --v  --max_mb_gpu 2000 --huber_icp 0.05 --icp_max_normal 45 --gn_max_iterations 30 --new_render --use_new_alignment --align_debug_images_scale 0.5'
command_string += ' --cl_path . --cli'
# this one must be last
command_string += ' --arun "%s" --output "%s" > "%s"'

if __name__=='__main__':
    if (len(sys.argv)) != 3:
        print "Usage: python %s [input_folder] [output_folder]" % this_filename_for_usage
        sys.exit(1)
    input_base_folder = os.path.abspath(sys.argv[1])
    output_base_folder = os.path.abspath(sys.argv[2])
    # just assume new structure of "two deep"
    object_path_list = [os.path.abspath(os.path.join(input_base_folder, x)) for x in os.listdir(input_base_folder)]
    for object_path in object_path_list:
        view_path_list = [os.path.abspath(os.path.join(object_path, x)) for x in os.listdir(object_path)]
        object_name = os.path.split(object_path)[1]
        for view_path in view_path_list:
            view_name = os.path.split(view_path)[1]
            input_full_path = os.path.abspath(view_path)
            output_folder_for_input_base = os.path.join(output_base_folder, object_name, view_name)
            if not os.path.exists(output_folder_for_input_base): os.makedirs(output_folder_for_input_base)
            command = command_string % (input_full_path, output_folder_for_input_base, output_folder_for_input_base + '.txt')
            print command
            subprocess.call(command, shell=True, cwd=cwd)


# attempt with os.walk:
if False:
    for dir_name, subdir_list, file_list in os.walk(input_base_folder):
        if 'color' in subdir_list:
            del subdir_list[:]
            print dir_name
            input_full_path = os.path.abspath(dir_name) # probably not necessary
            input_file_name = os.path.split(input_full_path)[1]
            output_folder_name = input_file_name
            output_folder_for_input_base = os.path.join(output_base_folder, output_folder_name)
            if not os.path.exists(output_folder_for_input_base): os.makedirs(output_folder_for_input_base)
            command = command_string % (input_full_path, output_folder_for_input_base, output_folder_for_input_base + '.txt')
            print command
            subprocess.call(command, shell=True, cwd=cwd)

# old stuff:
if False:
    input_folder_path_list = [os.path.abspath(os.path.join(input_base_folder, x)) for x in os.listdir(input_base_folder)]
    for input_full_path in input_folder_path_list:
        print input_full_path
        input_file_name = os.path.split(input_full_path)[1]
        output_folder_name = input_file_name
        output_folder_for_input_base = os.path.join(output_base_folder, output_folder_name)
        if not os.path.exists(output_folder_for_input_base): os.makedirs(output_folder_for_input_base)
        command = command_string % (input_full_path, output_folder_for_input_base, output_folder_for_input_base + '.txt')
        print command
        subprocess.call(command, shell=True, cwd=cwd)
