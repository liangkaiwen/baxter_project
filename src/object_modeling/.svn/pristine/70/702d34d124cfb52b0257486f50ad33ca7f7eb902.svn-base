#!/bin/python

import sys
import os
import os.path
import subprocess

evaluate_ate_py = 'C:/devlibs/rgbd_benchmark_tools/src/rgbd_benchmark_tools/evaluate_ate.py'
evaluate_rpe_py = 'C:/devlibs/rgbd_benchmark_tools/src/rgbd_benchmark_tools/evaluate_rpe.py'

if __name__=='__main__':
    ground_truth_base_folder = os.path.abspath(sys.argv[1])
    results_base_folder = os.path.abspath(sys.argv[2])

    abspaths = [os.path.abspath(os.path.join(results_base_folder, x)) for x in os.listdir(results_base_folder)]
    for result_folder in abspaths:
        if not os.path.isdir(result_folder): continue
        result_folder_name = os.path.split(result_folder)[1]
        expected_result_file = os.path.join(result_folder, 'mesh_final', 'camera_poses.txt')
        if (not os.path.exists(expected_result_file)):
            print 'warning: expected result file ' + expected_result_file
            continue
        expected_ground_truth_file = os.path.join(ground_truth_base_folder, result_folder_name, 'groundtruth.txt')
        if (not os.path.exists(expected_ground_truth_file)):
            print 'warning: expected ground truth file ' + expected_ground_truth_file
            continue
        
        print result_folder_name

        command = 'python %s %s %s' % (evaluate_ate_py, expected_ground_truth_file, expected_result_file)
        ate_output = subprocess.check_output(command, shell=True)
        print 'evaluate_ate_py:', ate_output.strip()
        
        do_rpe = False
        if do_rpe:
            command = 'python %s %s %s' % (evaluate_rpe_py, expected_ground_truth_file, expected_result_file)
            rpe_output = subprocess.check_output(command, shell=True)   
            print 'evaluate_rpe_py:', rpe_output.strip()