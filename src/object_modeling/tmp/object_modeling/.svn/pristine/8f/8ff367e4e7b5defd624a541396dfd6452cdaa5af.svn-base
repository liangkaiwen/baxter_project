#!/bin/python

import sys
import os
import os.path
import subprocess

this_filename_for_usage = 'luis_batch_folder.py'

# windows:
#cwd = None
#exe_file = 'C:/devlibs/object_modeling/batch_ready/volume_modeler_main'
# linux:
#cwd = '.'
cwd = '/home/peter/volume_modeler_release'
exe_file = os.path.join(cwd, 'volume_modeler_main')

command_string = exe_file
#command_string += ' --circle_mask_pixel_radius 300 --circle_mask_x 320 --circle_mask_y 240'
command_string += ' --rect_mask_x 180 --rect_mask_y 0 --rect_mask_width 280 --rect_mask_height 420'
command_string += ' --save_masked_input'
command_string += ' --use_f200_camera --turntable_rotation_limit 400    --frame_increment 1 --f --vs 0.001    --grid_size 64 --max 0.6  --mt grid --v  --max_mb_gpu 2000 --huber_icp 0.05 --icp_max_normal 45 --gn_max_iterations 30 --new_render --use_new_alignment --align_debug_images_scale 0.5'
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
        object_name = os.path.split(object_path)[1]
        input_full_path = os.path.abspath(object_path)
        output_folder_for_input_base = os.path.join(output_base_folder, object_name)
        if not os.path.exists(output_folder_for_input_base): os.makedirs(output_folder_for_input_base)
        command = command_string % (input_full_path, output_folder_for_input_base, output_folder_for_input_base + '.txt')
        print command
        subprocess.call(command, shell=True, cwd=cwd)


