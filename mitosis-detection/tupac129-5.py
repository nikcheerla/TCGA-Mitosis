from train_icpr import train_icpr

params_dict = {}

params_dict['net_name'] = 'detector129-5'

params_dict['size'] = 2084
params_dict['patch_size'] = 129
params_dict['radius'] = 10
params_dict['normalization'] = False

params_dict['N'] = 15000
params_dict['MN'] = 8000
params_dict['K'] = 2.5
params_dict['epochs'] = 15
params_dict['decay'] = 1.0
params_dict['kgrowth'] = 1.8
params_dict['egrowth'] = 0.97

train_icpr(params_dict)
