#!/bin/python

import sys
import os
import os.path
import subprocess

exe_file = 'C:/devlibs/object_modeling/batch_ready/volume_modeler_main'
command_string = exe_file
command_string += ' --cli --vs 0.0025 --grid_size 64 --max 1.5 --frame_increment 3 --f --mt grid --v  --max_mb_gpu 2000'
#command_string += ' --save_mesh_every_n 200 --save_state_every_n 200'
command_string += ' --oni "%s" --output "%s" > "%s"'
output_subfolder = 'baxter_single_mesh'

if __name__=='__main__':
    input_base_folder = os.path.abspath(sys.argv[1])
    output_base_folder = os.path.abspath(sys.argv[2])
    input_folder_path_list = [os.path.abspath(os.path.join(input_base_folder, x)) for x in os.listdir(input_base_folder)]
    for input_full_path in input_folder_path_list:
        print input_full_path
        input_file_name = os.path.split(input_full_path)[1]
        if '.oni' not in input_file_name: continue
        output_folder_name = input_file_name
        output_folder_for_input_base = os.path.join(output_base_folder, output_subfolder, output_folder_name)
        if not os.path.exists(output_folder_for_input_base): os.makedirs(output_folder_for_input_base)
        command = command_string % (input_full_path, output_folder_for_input_base, output_folder_for_input_base + '.txt')
        subprocess.call(command, shell=True)