from datetime import datetime

args = {
    "tag": "salivqa_dmos",
    "debug": False,
    "num_workers": 1,
    "random_seed": 1234,
    "device": "cuda",
    "time": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),

    "extractor_name": "DeformViTExtractor",

	"feature_norm": True,

	"model_config_pos": {
		"input_resolution": 224,
		"model_resolution": 224,
		"patch_size": 16,
		"num_heads": 12,
		"num_features": 768,
		"input_type": "pano",
		"num_classes": 0,
		"train_type": "clip"
	},

    "model_config_neg": {
        "input_resolution": 224,
        "model_resolution": 224,
        "patch_size": 16,
        "num_heads": 12,
        "num_features": 768,
        "input_type": "pano",
        "num_classes": 1000,
        "train_type": "imgn"
    },

    "sta_config": 
    {
        "model_name": "STADecoder",
        'ckpt_path': '../data/log/2026_01_23_06_58_15_yodaczsbvw_',
        'ckpt_name': '04_ckpt',

        "lr": 1e-5,
        "weight_decay": 1e-3,
        "num_frame": 5,
        "eval_start_epoch": 0,

        "use_exp_decay": False,
        "use_epoch_decay": False,
        "iterdecay_rate": 0.5,
        "epochdecay_rate": 0.99999,


        "model_config": {
            "drop_prob": 0.1,               
            "lambda_spatial_feat": 1.0,
            "lambda_temporal_feat": 2.0,
            "lambda_cls": 1e3,
            "mask_prob": 0.15,
            "use_residual": True,
            "normalize": True,

            "num_features": 768,
            "input_resolution": 224,
            "patch_size": 16,

            "margin_cls": 2,

            "coeff_cls": 1,
            "coeff_time": 1,
            "coeff_space": 1,

            "criterion_mfp": "mse",

            "spatial_adjacency": "spherical",
            "heatmap_geometry": "spherical",

            "cls_encoder_type": "mlp",

            "depth": 1,
            "sigma": 16,
            "verbose_mode": False
        }
    },
    # {
    #     'ckpt_path': '../data/log/2026_01_23_05_46_31_yodaczsbvw_',
    #     'ckpt_name': '04_ckpt',
    #     "model_name": "STADecoder",

    #     "lr": 1e-5,
    #     "weight_decay": 1e-3,
    #     "num_frame": 5,
    #     "eval_start_epoch": 0,

    #     "use_exp_decay": False,
    #     "use_epoch_decay": False,
    #     "iterdecay_rate": 0.5,
    #     "epochdecay_rate": 0.99999,


    #     "model_config": {
    #         "drop_prob": 0.1,               
    #         "lambda_spatial_feat": 1.0,
    #         "lambda_temporal_feat": 50.0,
    #         "lambda_cls": 1e5,
    #         "mask_prob": 0.15,
    #         "use_residual": True,
    #         "normalize": True,

    #         "num_features": 768,
    #         "input_resolution": 224,
    #         "patch_size": 16,

    #         "margin_cls": 2,

    #         "coeff_cls": 1,
    #         "coeff_time": 1,
    #         "coeff_space": 1,

    #         "criterion_mfp": "mse",

    #         "spatial_adjacency": "spherical",
    #         "heatmap_geometry": "spherical",

    #         "cls_encoder_type": "mlp",

    #         "depth": 1,
    #         "sigma": 16,
    #         "verbose_mode": False
    #     }
    # },


    # Dataset/model selection
    "data_config_path": "./code/configs/dataset/vqa_dmos.json",
    "model_config_path": "./code/configs/model/salivqa.json",
    "brisque_model_path": "/home/ivan/projects/heat/ViT-ODV/data/brisque/brisque_model_live.yml",
    "brisque_range_path": "/home/ivan/projects/heat/ViT-ODV/data/brisque/brisque_range_live.yml",


    # Data paths
    "data_path": "./data",
    "log_path": "./data/log",
    "cache_path": "./data/cache",
    "rebuild_cache": False,
    "list_path": "/home/ivan/projects/heat/ViT-ODV/data/VQA_ODV/train_dmos.txt",  # defaults to data_path/VQA_ODV/train_dmos.txt

    # Optional: SaliVQA checkpoint to resume training (train_salivqa.py does not load it yet)
    "ckpt_path": "/home/ivan/projects/heat/ViT-ODV/data/log/2026_01_30_08_43_31_yodaczsbvw_salivqa_dmos",
    "ckpt_name": "09_ckpt",

    # Training config (consumed by train_salivqa.py)

    "save_model": True,

    "optimizer_name": "Adam",
    "scheduler_name": "none",

    "display_config": {
        "concat": True,
        "overlay": False,
    },
}
