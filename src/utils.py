import numpy as np
import time
import math
import matplotlib.pyplot as plt

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=1) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
def plot_losses(train_loss, val_loss, scale):
    plt.figure(figsize=(10,5))
    plt.plot(train_loss)
    plt.plot([(x + 1) * scale - 1 for x in range(len(val_loss))], val_loss)
    plt.legend(['train loss', 'validation loss'])

def get_avg_length(verb, pairs):
    total_tokens = []
    for ix_pair, pair in enumerate(pairs):
        if verb in pair[0].split():
            total_tokens.append(len(pair[0].split()))
    
    total_tokens = np.array(total_tokens)
    return total_tokens.mean(), total_tokens.std(), len(total_tokens)

def get_stats(report, train_pairs, test_pairs):
    
    r = {}
    for key in report:
        r[key] = {}
        r[key]['precision'] = (report[key]['hint'] / report[key]['total_out_ambiguous']) if  report[key]['total_out_ambiguous'] else report[key]['total_out_ambiguous']
        r[key]['coverage'] = report[key]['hint'] / report[key]['total_in_ambiguous']
        
        temp_mean, temp_std, temp_total = get_avg_length(key, train_pairs)
        r[key]['train_avg_lenght'] = temp_mean
        r[key]['train_std_lenght'] = temp_std
        r[key]['train_n'] = temp_total
        temp_mean, temp_std, temp_total = get_avg_length(key, test_pairs)
        r[key]['test_avg_lenght'] = temp_mean
        r[key]['test_std_lenght'] = temp_std
        r[key]['test_n'] = temp_total   
        
    res = pd.DataFrame(r).transpose().sort_values(by='precision')
    return res