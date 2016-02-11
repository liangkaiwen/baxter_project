import numpy as np
import matplotlib.pyplot as plt
import sys
import collections

def get_values_from_run(filename):
    lines = open(filename)
    values = collections.defaultdict(list)
    for line in lines:
        if 'TIME entire frame:' in line: values['frame'].append(float(line.split()[-1]))
        if 'TIME render:' in line: values['render'].append(float(line.split()[-1]))
        if 'TIME dense alignment:' in line: values['dense'].append(float(line.split()[-1]))
        if 'TIME add frame:' in line: values['add'].append(float(line.split()[-1]))
        if 'TIME update_interface_' in line: values['interface'].append(float(line.split()[-1]))
        if 'TIME alignAndAddFrame:' in line: values['align_add'].append(float(line.split()[-1]))
        
        if 'result_iterations' in line and 'loop' not in line:
            this_iters = [int(x) for x in line.split()[1:]]
            values['iterations'].append(this_iters)
    
    # do some numpy cleanup.
    values['frame'] = np.array(values['frame'][1:])
    values['render'] = np.array(values['render'])
    values['dense'] = np.array(values['dense'])
    values['add'] = np.array(values['add'][1:])
    values['interface'] = np.array(values['interface'][1:])
    values['align_add'] = np.array(values['align_add'][1:])
    values['known_sum'] = values['render'] + values['dense'] + values['add'] + values['interface']
    values['other'] = values['frame'] - values['known_sum']
    
    return values
    

if __name__ == "__main__":
    #filename = sys.argv[1]
    #filename = "run.txt"
    filename = "one_volume_multiscale.txt"
    values = get_values_from_run(filename)
    #values_2 = get_values_from_run("run_single_fs3_gn40.txt")
    
    x_axis = xrange(len(values['frame']))
    
    plt.clf()
    save_name = None
    if True:
        save_name = filename + '.stackplot.svg'
        stackplot_result = plt.stackplot(x_axis, values['dense'], values['add'], values['render'], values['interface'], values['other'])
        proxy_rects = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in stackplot_result]
        label_list = ['dense', 'add', 'render', 'interface', 'other']
        plt.legend(proxy_rects, label_list)
    if False:
        save_name = filename + '.iterations.svg'
        plt.plot(values['iterations'])
    if False:
        save_name = filename + '.align_add.svg'
        p1, = plt.plot(values['align_add'])
        p2, = plt.plot(values['known_sum'])
        p3, = plt.plot(values['frame'])
        plt.legend([p1,p2,p3], ['align_add', 'known_sum', 'frame'])
    if False:
        save_name = filename + '.compare.svg'
        p1, = plt.plot(values['dense'])
        p2, = plt.plot(values_2['dense'])
        plt.legend([p1,p2], ['multiscale', 'single_scale_40'], loc=2 )
    plt.savefig(save_name)
    print "saved figure:",save_name
    plt.show()