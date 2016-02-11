
import sys
import os
import os.path
import subprocess

# intel 5th floor for evan:
# These were old:
#command_string = '..\\bin\\script --script --save-poses --of --md --vn 512 --vs 0.005 --oc --ocln --max 2.0 --input "%s" --output "%s"'
#command_string = '..\\bin\\script --script --save-poses --of --md --vn 512 --vs 0.02 --oc --ocln --max 5.0 --input "%s" --output "%s"'

#exe_file = '..\\Release\\object_model --script'
exe_file = '/home/peter/checkout/object_modeling/build/object_model --script'
command_string_base = exe_file
command_string_base += ' --save-poses --input "%s" --output "%s"'
command_string_base += ' --oc --oclpath "/home/peter/checkout/object_modeling/OpenCLStaticLib"'
# remote desktop (windows at least) disables nvidia:
command_string_base += ' --ocln'
# these are old and not needed for evan:
#command_string_base += ' --pv-be 0.2   --pv-loop-di --pv-debug-vis  --pv-loop-features  --pv-loop-min-coverage "0.5" --pv-loop-skip 10 --pv-max-side 128  --pv-edge-loop-factor 100 --pv-max-age-loop 200'
command_string_base += ' --show-axes --pv-show-graph-edges --pv-show-mesh'
command_string_base += ' --c-show-render --cdis 0.5'
#command_string_base += ' --vs 0.005 --fonipng --max 1.2 --fi 3 --fe 450'   # SEE HOW BROKEN THIS IS
#command_string_base += ' --frgbd --minput --mo'
add_for_loop = ' --pv --pv-loop'
#add_for_single = ' --vn 256'
# evan style:
#add_for_single = ' --of --fff --vn 512 --vs 0.02 --max 5.0'
# FIXED evan style:
#add_for_single = ' --of --fff --vn 512 --vs 0.005 --max 2.0'
# block evan style
#add_for_single = ' --of --fff --vs 0.005 --max 2.0 --pv --pg --pg-border 0 --disable-viewer'
# newer block evan style (for iros)
#add_for_single = ' --of --fff --vs 0.005 --max 3.0 --pv --pg --pg-border 1 --disable-viewer'
# jaech crashed, try max 5.0, vs 0.01, max-mb 
add_for_single = ' --of --fff --vs 0.01 --max 5.0 --pv --pg --pg-border 1 --pv-max-mb 900 --disable-viewer'
add_for_load_poses = ' --load-object-poses --input-file-object-poses "%s"'
add_for_save_objects_pcd = ' --save-objects-pcd'
add_for_save_objects_png = ' --save-objects-png'
add_for_log = ' > "%s"' # needs to make sure location exists....


if __name__=='__main__':
    do_loop = False
    do_single = True
    do_save_pcd_single = False
    do_save_png_single = False

    input_base_folder = os.path.abspath(sys.argv[1])
    output_base_folder = os.path.abspath(sys.argv[2])
    for input_folder in [os.path.abspath(os.path.join(input_base_folder, x)) for x in os.listdir(input_base_folder)]:
        output_folder_for_input_base = os.path.join(output_base_folder, os.path.split(input_folder)[1] + "-output")
        # create at least the base path so we can dump logs there
        if not os.path.exists(output_folder_for_input_base): os.makedirs(output_folder_for_input_base)

        if (do_loop):
            output_folder = os.path.join(output_folder_for_input_base, "loop")
            command = (command_string_base % (input_folder, output_folder)) + add_for_loop + (add_for_log % os.path.join(output_folder_for_input_base, "loop.log"))
            subprocess.call(command, shell=True)
            # SEE THAT YOU CHANGED THIS
            #expected_poses_file = os.path.join(output_folder, "object_poses_loop.txt")
            #mesh_output_folder = os.path.join(output_folder_for_input_base, "loop_mesh")
            expected_poses_file = os.path.join(output_folder, "object_poses_sequential.txt")
            mesh_output_folder = os.path.join(output_folder_for_input_base, "sequential_mesh")
            command =  (command_string_base % (input_folder, mesh_output_folder)) + add_for_single + (add_for_load_poses % expected_poses_file) + (add_for_log % os.path.join(output_folder_for_input_base, "loop_mesh.log"))
            subprocess.call(command, shell=True)

        if (do_single):
            output_folder = os.path.join(output_folder_for_input_base, "single")
            command = (command_string_base % (input_folder, output_folder)) + add_for_single + (add_for_log % os.path.join(output_folder_for_input_base, "single.log"))
            if (do_save_pcd_single):
                command += add_for_save_objects_pcd
            if (do_save_png_single):
                command += add_for_save_objects_png
            print command
            subprocess.call(command, shell=True)

# old single way to do it:
        """
        command = command_string % (input_folder, output_folder)
        print command
        subprocess.call(command, shell=True)
        """