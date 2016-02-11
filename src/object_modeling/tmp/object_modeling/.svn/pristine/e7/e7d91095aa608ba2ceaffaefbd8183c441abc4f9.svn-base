#!/bin/python

import sys
import os
import os.path
import subprocess

#exe_file = 'C:/devlibs/object_modeling/batch_ready/volume_modeler_main'
#exe_file = 'C:/devlibs/object_modeling/build/Release/volume_modeler_main'
cwd = '/home/peter/checkout/object_modeling/build/'
exe_file = '/home/peter/checkout/object_modeling/build/volume_modeler_main'
command_string = exe_file
command_string += ' --cli --vs 0.0025 --vxyz 256 --first_frame_origin --v'
command_string += ' --mt single --mt1 kmeans'
command_string += ' --normals_smooth_iterations 10 --use_six_volumes --compatibility_add_max_angle_degrees 45 --cos_weight'
command_string += ' --min_truncation_distance 0.02'
command_string += ' --cl_path .'
command_string += ' --png_depth "%s" --output "%s" > "%s"'

if __name__=='__main__':
    input_base_folder = os.path.abspath(sys.argv[1])
    output_base_folder = os.path.abspath(sys.argv[2])
    input_folder_path_list = [os.path.abspath(os.path.join(input_base_folder, x)) for x in os.listdir(input_base_folder)]
    for input_full_path in input_folder_path_list:
        print input_full_path
        input_file_name = os.path.split(input_full_path)[1]
        output_folder_name = input_file_name
        output_folder_for_input_base = os.path.join(output_base_folder, output_folder_name)
        if not os.path.exists(output_folder_for_input_base): os.makedirs(output_folder_for_input_base)
        command = command_string % (input_full_path, output_folder_for_input_base, output_folder_for_input_base + '.txt')
        subprocess.call(command, shell=True, cwd=cwd)
