#!/bin/python

import sys
import os
import os.path
import subprocess

exe_file = 'C:/devlibs/object_modeling/batch_ready/volume_modeler_main'
command_string = exe_file
command_string += ' --cli --max 5 --mt grid --max_mb_gpu 2000 --v --f'
#command_string += ' --save_mesh_every_n 200 --save_state_every_n 200'
command_string += ' --freiburg "%s" --output "%s" > "%s"'
output_subfolder = 'freiburg_f_5m'

if __name__=='__main__':
    input_base_folder = os.path.abspath(sys.argv[1])
    output_base_folder = os.path.abspath(sys.argv[2])
    folders = [os.path.abspath(os.path.join(input_base_folder, x)) for x in os.listdir(input_base_folder) if 'rgbd' in x]
    for input_folder in folders:
        print input_folder
        input_folder_name = os.path.split(input_folder)[1]
        output_folder_name = input_folder_name
        output_folder_for_input_base = os.path.join(output_base_folder, output_subfolder, output_folder_name)
        if not os.path.exists(output_folder_for_input_base): os.makedirs(output_folder_for_input_base)
        command = command_string % (input_folder + '/associate.default.txt', output_folder_for_input_base, output_folder_for_input_base + '.txt')
        subprocess.call(command, shell=True)