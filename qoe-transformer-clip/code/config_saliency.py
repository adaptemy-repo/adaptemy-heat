from datetime import datetime
from munch import Munch

args = {
    'tag': '',
    'debug': False,
    'num_workers': 8,
    'random_seed': 1234,
    'device': 'cuda',
    'time': datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),

    'extractor_config_path': './code/configs/extractor/adaptvit.json', #'./code/configs/extractor/deformvit.json',
    'data_config_path': './code/configs/dataset/vqa_odv.json', #'./code/configs/dataset/wild360.json',
    'model_config_path': './code/configs/model/stadecoder_single_imgn.json', #'./code/configs/model/paver.json',

    'data_path': './data',
    'log_path': './data/log',                       # Overridden with local config
    'ckpt_path': './data/log/2026_01_23_06_58_15_yodaczsbvw_',
    # 'ckpt_path': './data/log',
    'cache_path': './data/cache',
    'rebuild_cache': False,
    'ckpt_name': '04_ckpt',
    # 'ckpt_name': None,
    'input_video': '/home/ivan/projects/heat/ViT-ODV/data/VQA_ODV/data/test/G4BikingInSaalbach_ERP_5600x2800_fps29.97_qp27_59876k.mp4',

    'save_model': True,

    'optimizer_name': 'Adam',
    'scheduler_name': 'none',

    'display_config': {
        'concat': True, # Display gt, prop, video at once for brevity
        'overlay': False
    }
}

# args = {
#     'tag': '',
#     'debug': False,
#     'num_workers': 8,
#     'random_seed': 1234,
#     'device': 'cuda',
#     'time': datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),

#     'extractor_config_path': './code/configs/extractor/adaptvit_single.json', 
#     'data_config_path': './code/configs/dataset/vqa_odv_single.json', 
#     'model_config_path': './code/configs/model/stadecoder_single.json', 

#     'data_path': './data',
#     'log_path': './data/log',                       # Overridden with local config
#     'ckpt_path': './data/log',
#     'cache_path': './data/cache',
#     'rebuild_cache': False,
#     'ckpt_name': None,
#     'input_video': '/home/ivan/projects/heat/ViT-ODV/data/VQA_ODV/data/train/G1LandSalt_ERP_6144x3072_fps29.97_qp27_9229k.mp4',

#     'save_model': True,

#     'optimizer_name': 'Adam',
#     'scheduler_name': 'none',

#     'display_config': {
#         'concat': True, # Display gt, prop, video at once for brevity
#         'overlay': False
#     }
# }