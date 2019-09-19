from os.path import join

def read_file(path):
    with open(path) as file:
        content = file.read()       
    return content 

def parse_weights(weights_str):
    weights_str = weights_str.split('\n')
    manage_weights = {}
    out_weights = {}
    
    for el in weights_str:
        w_name, w_value = el.split('=')
        if w_name.startswith('a'):
            out_weights[int(w_name[1:])] = float(w_value)
        elif w_name.startswith('b'):
            manage_weights[int(w_name[1:])] = float(w_value)
        else:
            raise ValueError('Unresolved signal')
            
    manage_weights = {i:manage_weights.get(i,0.) for i in range(max(manage_weights.keys())+1)}
    out_weights = {i:out_weights.get(i,0.) for i in range(max(out_weights.keys())+1)}
            
    return manage_weights, out_weights

def parse_signals(signal_str):
    return list(map(float , signal_str.split('\n')))

def parse_file(path):
    real_weights = read_file(join(path,'test.txt'))
    manage_signals = read_file(join(path,'v.txt'))
    out_signals = read_file(join(path,'y.txt'))
    
    manage_weights, out_weights = parse_weights(real_weights)
    manage_weights[0] = 1.
    
    manage_signals = parse_signals(manage_signals)
    out_signals = parse_signals(out_signals)
    
    return {
        'manage_weights':manage_weights,
        'manage_signals':manage_signals,
        'out_weights':out_weights,
        'out_signals':out_signals
    }