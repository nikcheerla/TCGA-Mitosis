from train_icpr import train_icpr

params_dict = {}

params_dict['net_name'] = 'detector129-1'


params_dict['size'] = 2084
params_dict['patch_size'] = 129
params_dict['radius'] = 10
params_dict['normalization'] = False

params_dict['N'] = 12000
params_dict['MN'] = 12000
params_dict['K'] = 1.5
params_dict['epochs'] = 15
params_dict['decay'] = 1.0
params_dict['kgrowth'] = 1.8
params_dict['egrowth'] = 0.95

train_icpr(params_dict)