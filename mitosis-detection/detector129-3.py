from train_icpr import train_icpr

params_dict = {}

params_dict['net_name'] = 'detector129-3'


params_dict['size'] = 2084
params_dict['patch_size'] = 129
params_dict['radius'] = 10
params_dict['normalization'] = False

params_dict['N'] = 10000
params_dict['MN'] = 10000
params_dict['K'] = 0.5
params_dict['epochs'] = 7
params_dict['decay'] = 1.0
params_dict['kgrowth'] = 1.3
params_dict['egrowth'] = 1.01

train_icpr(params_dict)
