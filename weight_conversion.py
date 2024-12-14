import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from ops_dcnv3 import modules as opsm
from config_intern_g import config_dict
from mapping import layer_name_mapping_dict

import os
import sys
models_path = os.path.join(os.getcwd(), "submodules", "internimage", "classification", "models")
dcn_path = os.path.join(os.getcwd(), "submodules", "internimage", "classification", )
sys.path.append(models_path)
sys.path.append(dcn_path)

from models.intern_image import InternImage
from ops_dcnv3 import modules as opsm
from model import InternImageCustom

def build_model(config):
    model_type = config["MODEL"]["TYPE"]
    if model_type == 'intern_image':
        model = InternImage(
            core_op=config["MODEL"]["INTERN_IMAGE"]["CORE_OP"],
            num_classes=config["MODEL"]["NUM_CLASSES"],
            channels=config["MODEL"]["INTERN_IMAGE"]["CHANNELS"],
            depths=config["MODEL"]["INTERN_IMAGE"]["DEPTHS"],
            groups=config["MODEL"]["INTERN_IMAGE"]["GROUPS"],
            layer_scale=config["MODEL"]["INTERN_IMAGE"]["LAYER_SCALE"],
            offset_scale=config["MODEL"]["INTERN_IMAGE"]["OFFSET_SCALE"],
            post_norm=config["MODEL"]["INTERN_IMAGE"]["POST_NORM"],
            mlp_ratio=config["MODEL"]["INTERN_IMAGE"]["MLP_RATIO"],
            with_cp=config["TRAIN"]["USE_CHECKPOINT"],
            drop_path_rate=config["MODEL"]["DROP_PATH_RATE"],
            res_post_norm=config["MODEL"]["INTERN_IMAGE"]["RES_POST_NORM"], # for InternImage-H/G
            dw_kernel_size=config["MODEL"]["INTERN_IMAGE"]["DW_KERNEL_SIZE"], # for InternImage-H/G
            use_clip_projector=config["MODEL"]["INTERN_IMAGE"]["USE_CLIP_PROJECTOR"], # for InternImage-H/G
            level2_post_norm=config["MODEL"]["INTERN_IMAGE"]["LEVEL2_POST_NORM"], # for InternImage-H/G
            level2_post_norm_block_ids=config["MODEL"]["INTERN_IMAGE"]["LEVEL2_POST_NORM_BLOCK_IDS"], # for InternImage-H/G
            center_feature_scale=config["MODEL"]["INTERN_IMAGE"]["CENTER_FEATURE_SCALE"], # for InternImage-H/G
            remove_center=config["MODEL"]["INTERN_IMAGE"]["REMOVE_CENTER"],
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

og_model = build_model(config_dict)
og_model_ckpt = torch.load("internimage_g_22kto1k_512.pth", weights_only=True, map_location='cuda')

for original_name, my_name in layer_name_mapping_dict.items():
    if original_name in og_model_ckpt["model"]:
        pretrained_tensor = og_model_ckpt["model"][original_name]
        print('here')
      
      # Example of navigating to a nested parameter in your model
        module_names = my_name.split(".")
        for name in module_names[:-1]:
            module = getattr(module, name)
        param = getattr(module, module_names[-1])  # Get the parameter

      # Shape check
        if param.shape == pretrained_tensor.shape:
            with torch.no_grad():
                print("here 2")
                param.copy_(pretrained_tensor)
        else:
            print(f"Shape mismatch for {my_name}: "
                f"Expected {param.shape}, got {pretrained_tensor.shape}")
    else:
        print(f"Parameter {original_name} not found in pre-trained weights.")

own_model = InternImageCustom()

own_model.save("internimage_g_22kto1k_512.pth")