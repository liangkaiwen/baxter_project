#import matplotlib.pyplot as plt
#import numpy as np
from matplotlib.pylab import *
import os.path

folders_to_compare = ['cascade_2']
files_to_load = [('error_used', 'tov-color-error-diff-relative.txt'),
                 ('nonzero', 'tov-nonzero-count.txt'),
                 ('success', 'tov-icp-success.txt'), 
                 ('e_color', 'tov-error-color.txt'), 
                 ('p_e_color', 'tov-previous-error-color.txt'),
                 ('ev_min_before', 'tov-eigen-min-before.txt'),
                 ('ev_min_after', 'tov-eigen-min-after.txt'),
                 ('rank_after', 'tov-rank-after.txt')
                 ]
d = {}

w_diff = array([1.0, -1.0])
def my_diff(data):
    return convolve(data, w_diff)[:-1]

def sliding_mean(data, n):
    weights = repeat(1.0, n) / n
    return convolve(data, weights)[:-(n-1)]

if __name__=='__main__':
    for folder in folders_to_compare:
        d[folder] = {}
        for name, filename in files_to_load:
            fullpath = (os.path.join(folder, filename))
            d[folder][name] = loadtxt(fullpath)

    for folder, item in d.iteritems():
        label_pre = folder + ':'
        error_used = item['error_used']
        plot(error_used, label=label_pre + 'error_used')
        error_color = item['e_color']
        plot(error_color, label=label_pre + 'error_color_raw')
        nonzero = item['nonzero']
        #plot(nonzero, label=label_pre + 'nonzero')
        success = item['success']
        plot(success, label=label_pre + 'success')
        #d = item['e_color']
        #plot(d, label=label_pre + 'e_color')
        #d = item['p_e_color']
        #plot(d, label=label_pre + 'p_e_color')
        ev_after = item['ev_min_after']
        #plot(ev_after, label=label_pre + 'ev_after')
        rank_after = item['rank_after']
        #plot(rank_after, label=label_pre + 'rank_after')
        sqrt_ev_after = sqrt(ev_after)
        #plot(sqrt_ev_after, label=label_pre + 'sqrt_ev_after')
        
    legend()
    show()