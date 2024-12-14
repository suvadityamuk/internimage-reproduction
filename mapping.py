layer_name_mapping_dict = {
    'patch_embed.norm1.1.weight': 'patch_embed.norm1.weight',
    'patch_embed.norm1.1.bias': 'patch_embed.norm1.bias',
    'patch_embed.norm2.1.weight': 'patch_embed.norm2.weight',
    'patch_embed.norm2.1.bias': 'patch_embed.norm2.bias',
    'levels.0.blocks.0.norm1.0.weight': 'levels.0.blocks.0.norm1.weight',
    'levels.0.blocks.0.norm1.0.bias': 'levels.0.blocks.0.norm1.bias',
    'levels.0.blocks.0.norm2.0.weight': 'levels.0.blocks.0.norm2.weight',
    'levels.0.blocks.0.norm2.0.bias': 'levels.0.blocks.0.norm2.bias',
    'levels.0.blocks.1.norm1.0.weight': 'levels.0.blocks.1.norm1.weight',
    'levels.0.blocks.1.norm1.0.bias': 'levels.0.blocks.1.norm1.bias',
    'levels.0.blocks.1.norm2.0.weight': 'levels.0.blocks.1.norm2.weight',
    'levels.0.blocks.1.norm2.0.bias': 'levels.0.blocks.1.norm2.bias',
    'levels.0.norm.0.weight': 'levels.0.norm.weight',
    'levels.0.norm.0.bias': 'levels.0.norm.bias',
    'levels.0.downsample.norm.1.weight': 'levels.0.downsample.norm.weight',
    'levels.0.downsample.norm.1.bias': 'levels.0.downsample.norm.bias',
    'levels.1.blocks.0.norm1.0.weight': 'levels.1.blocks.0.norm1.weight',
    'levels.1.blocks.0.norm1.0.bias': 'levels.1.blocks.0.norm1.bias',
    'levels.1.blocks.0.norm2.0.weight': 'levels.1.blocks.0.norm2.weight',
    'levels.1.blocks.0.norm2.0.bias': 'levels.1.blocks.0.norm2.bias',
    'levels.1.blocks.1.norm1.0.weight': 'levels.1.blocks.1.norm1.weight',
    'levels.1.blocks.1.norm1.0.bias': 'levels.1.blocks.1.norm1.bias',
    'levels.1.blocks.1.norm2.0.weight': 'levels.1.blocks.1.norm2.weight',
    'levels.1.blocks.1.norm2.0.bias': 'levels.1.blocks.1.norm2.bias',
    'levels.1.norm.0.weight': 'levels.1.norm.weight',
    'levels.1.norm.0.bias': 'levels.1.norm.bias',
    'levels.1.downsample.norm.1.weight': 'levels.1.downsample.norm.weight',
    'levels.1.downsample.norm.1.bias': 'levels.1.downsample.norm.bias',
    'levels.2.blocks.0.norm1.0.weight': 'levels.2.blocks.0.norm1.weight',
    'levels.2.blocks.0.norm1.0.bias': 'levels.2.blocks.0.norm1.bias',
    'levels.2.blocks.0.norm2.0.weight': 'levels.2.blocks.0.norm2.weight',
    'levels.2.blocks.0.norm2.0.bias': 'levels.2.blocks.0.norm2.bias',
    'levels.2.blocks.1.norm1.0.weight': 'levels.2.blocks.1.norm1.weight',
    'levels.2.blocks.1.norm1.0.bias': 'levels.2.blocks.1.norm1.bias',
    'levels.2.blocks.1.norm2.0.weight': 'levels.2.blocks.1.norm2.weight',
    'levels.2.blocks.1.norm2.0.bias': 'levels.2.blocks.1.norm2.bias',
    'levels.2.blocks.2.norm1.0.weight': 'levels.2.blocks.2.norm1.weight',
    'levels.2.blocks.2.norm1.0.bias': 'levels.2.blocks.2.norm1.bias',
    'levels.2.blocks.2.norm2.0.weight': 'levels.2.blocks.2.norm2.weight',
    'levels.2.blocks.2.norm2.0.bias': 'levels.2.blocks.2.norm2.bias',
    'levels.2.blocks.3.norm1.0.weight': 'levels.2.blocks.3.norm1.weight',
    'levels.2.blocks.3.norm1.0.bias': 'levels.2.blocks.3.norm1.bias',
    'levels.2.blocks.3.norm2.0.weight': 'levels.2.blocks.3.norm2.weight',
    'levels.2.blocks.3.norm2.0.bias': 'levels.2.blocks.3.norm2.bias',
    'levels.2.blocks.4.norm1.0.weight': 'levels.2.blocks.4.norm1.weight',
    'levels.2.blocks.4.norm1.0.bias': 'levels.2.blocks.4.norm1.bias',
    'levels.2.blocks.4.norm2.0.weight': 'levels.2.blocks.4.norm2.weight',
    'levels.2.blocks.4.norm2.0.bias': 'levels.2.blocks.4.norm2.bias',
    'levels.2.blocks.5.norm1.0.weight': 'levels.2.blocks.5.norm1.weight',
    'levels.2.blocks.5.norm1.0.bias': 'levels.2.blocks.5.norm1.bias',
    'levels.2.blocks.5.norm2.0.weight': 'levels.2.blocks.5.norm2.weight',
    'levels.2.blocks.5.norm2.0.bias': 'levels.2.blocks.5.norm2.bias',
    'levels.2.blocks.6.norm1.0.weight': 'levels.2.blocks.6.norm1.weight',
    'levels.2.blocks.6.norm1.0.bias': 'levels.2.blocks.6.norm1.bias',
    'levels.2.blocks.6.norm2.0.weight': 'levels.2.blocks.6.norm2.weight',
    'levels.2.blocks.6.norm2.0.bias': 'levels.2.blocks.6.norm2.bias',
    'levels.2.blocks.7.norm1.0.weight': 'levels.2.blocks.7.norm1.weight',
    'levels.2.blocks.7.norm1.0.bias': 'levels.2.blocks.7.norm1.bias',
    'levels.2.blocks.7.norm2.0.weight': 'levels.2.blocks.7.norm2.weight',
    'levels.2.blocks.7.norm2.0.bias': 'levels.2.blocks.7.norm2.bias',
    'levels.2.blocks.8.norm1.0.weight': 'levels.2.blocks.8.norm1.weight',
    'levels.2.blocks.8.norm1.0.bias': 'levels.2.blocks.8.norm1.bias',
    'levels.2.blocks.8.norm2.0.weight': 'levels.2.blocks.8.norm2.weight',
    'levels.2.blocks.8.norm2.0.bias': 'levels.2.blocks.8.norm2.bias',
    'levels.2.blocks.9.norm1.0.weight': 'levels.2.blocks.9.norm1.weight',
    'levels.2.blocks.9.norm1.0.bias': 'levels.2.blocks.9.norm1.bias',
    'levels.2.blocks.9.norm2.0.weight': 'levels.2.blocks.9.norm2.weight',
    'levels.2.blocks.9.norm2.0.bias': 'levels.2.blocks.9.norm2.bias',
    'levels.2.blocks.10.norm1.0.weight': 'levels.2.blocks.10.norm1.weight',
    'levels.2.blocks.10.norm1.0.bias': 'levels.2.blocks.10.norm1.bias',
    'levels.2.blocks.10.norm2.0.weight': 'levels.2.blocks.10.norm2.weight',
    'levels.2.blocks.10.norm2.0.bias': 'levels.2.blocks.10.norm2.bias',
    'levels.2.blocks.11.norm1.0.weight': 'levels.2.blocks.11.norm1.weight',
    'levels.2.blocks.11.norm1.0.bias': 'levels.2.blocks.11.norm1.bias',
    'levels.2.blocks.11.norm2.0.weight': 'levels.2.blocks.11.norm2.weight',
    'levels.2.blocks.11.norm2.0.bias': 'levels.2.blocks.11.norm2.bias',
    'levels.2.blocks.12.norm1.0.weight': 'levels.2.blocks.12.norm1.weight',
    'levels.2.blocks.12.norm1.0.bias': 'levels.2.blocks.12.norm1.bias',
    'levels.2.blocks.12.norm2.0.weight': 'levels.2.blocks.12.norm2.weight',
    'levels.2.blocks.12.norm2.0.bias': 'levels.2.blocks.12.norm2.bias',
    'levels.2.blocks.13.norm1.0.weight': 'levels.2.blocks.13.norm1.weight',
    'levels.2.blocks.13.norm1.0.bias': 'levels.2.blocks.13.norm1.bias',
    'levels.2.blocks.13.norm2.0.weight': 'levels.2.blocks.13.norm2.weight',
    'levels.2.blocks.13.norm2.0.bias': 'levels.2.blocks.13.norm2.bias',
    'levels.2.blocks.14.norm1.0.weight': 'levels.2.blocks.14.norm1.weight',
    'levels.2.blocks.14.norm1.0.bias': 'levels.2.blocks.14.norm1.bias',
    'levels.2.blocks.14.norm2.0.weight': 'levels.2.blocks.14.norm2.weight',
    'levels.2.blocks.14.norm2.0.bias': 'levels.2.blocks.14.norm2.bias',
    'levels.2.blocks.15.norm1.0.weight': 'levels.2.blocks.15.norm1.weight',
    'levels.2.blocks.15.norm1.0.bias': 'levels.2.blocks.15.norm1.bias',
    'levels.2.blocks.15.norm2.0.weight': 'levels.2.blocks.15.norm2.weight',
    'levels.2.blocks.15.norm2.0.bias': 'levels.2.blocks.15.norm2.bias',
    'levels.2.blocks.16.norm1.0.weight': 'levels.2.blocks.16.norm1.weight',
    'levels.2.blocks.16.norm1.0.bias': 'levels.2.blocks.16.norm1.bias',
    'levels.2.blocks.16.norm2.0.weight': 'levels.2.blocks.16.norm2.weight',
    'levels.2.blocks.16.norm2.0.bias': 'levels.2.blocks.16.norm2.bias',
    'levels.2.blocks.17.norm1.0.weight': 'levels.2.blocks.17.norm1.weight',
    'levels.2.blocks.17.norm1.0.bias': 'levels.2.blocks.17.norm1.bias',
    'levels.2.blocks.17.norm2.0.weight': 'levels.2.blocks.17.norm2.weight',
    'levels.2.blocks.17.norm2.0.bias': 'levels.2.blocks.17.norm2.bias',
    'levels.2.blocks.18.norm1.0.weight': 'levels.2.blocks.18.norm1.weight',
    'levels.2.blocks.18.norm1.0.bias': 'levels.2.blocks.18.norm1.bias',
    'levels.2.blocks.18.norm2.0.weight': 'levels.2.blocks.18.norm2.weight',
    'levels.2.blocks.18.norm2.0.bias': 'levels.2.blocks.18.norm2.bias',
    'levels.2.blocks.19.norm1.0.weight': 'levels.2.blocks.19.norm1.weight',
    'levels.2.blocks.19.norm1.0.bias': 'levels.2.blocks.19.norm1.bias',
    'levels.2.blocks.19.norm2.0.weight': 'levels.2.blocks.19.norm2.weight',
    'levels.2.blocks.19.norm2.0.bias': 'levels.2.blocks.19.norm2.bias',
    'levels.2.blocks.20.norm1.0.weight': 'levels.2.blocks.20.norm1.weight',
    'levels.2.blocks.20.norm1.0.bias': 'levels.2.blocks.20.norm1.bias',
    'levels.2.blocks.20.norm2.0.weight': 'levels.2.blocks.20.norm2.weight',
    'levels.2.blocks.20.norm2.0.bias': 'levels.2.blocks.20.norm2.bias',
    'levels.2.blocks.21.norm1.0.weight': 'levels.2.blocks.21.norm1.weight',
    'levels.2.blocks.21.norm1.0.bias': 'levels.2.blocks.21.norm1.bias',
    'levels.2.blocks.21.norm2.0.weight': 'levels.2.blocks.21.norm2.weight',
    'levels.2.blocks.21.norm2.0.bias': 'levels.2.blocks.21.norm2.bias',
    'levels.2.blocks.22.norm1.0.weight': 'levels.2.blocks.22.norm1.weight',
    'levels.2.blocks.22.norm1.0.bias': 'levels.2.blocks.22.norm1.bias',
    'levels.2.blocks.22.norm2.0.weight': 'levels.2.blocks.22.norm2.weight',
    'levels.2.blocks.22.norm2.0.bias': 'levels.2.blocks.22.norm2.bias',
    'levels.2.blocks.23.norm1.0.weight': 'levels.2.blocks.23.norm1.weight',
    'levels.2.blocks.23.norm1.0.bias': 'levels.2.blocks.23.norm1.bias',
    'levels.2.blocks.23.norm2.0.weight': 'levels.2.blocks.23.norm2.weight',
    'levels.2.blocks.23.norm2.0.bias': 'levels.2.blocks.23.norm2.bias',
    'levels.2.blocks.24.norm1.0.weight': 'levels.2.blocks.24.norm1.weight',
    'levels.2.blocks.24.norm1.0.bias': 'levels.2.blocks.24.norm1.bias',
    'levels.2.blocks.24.norm2.0.weight': 'levels.2.blocks.24.norm2.weight',
    'levels.2.blocks.24.norm2.0.bias': 'levels.2.blocks.24.norm2.bias',
    'levels.2.blocks.25.norm1.0.weight': 'levels.2.blocks.25.norm1.weight',
    'levels.2.blocks.25.norm1.0.bias': 'levels.2.blocks.25.norm1.bias',
    'levels.2.blocks.25.norm2.0.weight': 'levels.2.blocks.25.norm2.weight',
    'levels.2.blocks.25.norm2.0.bias': 'levels.2.blocks.25.norm2.bias',
    'levels.2.blocks.26.norm1.0.weight': 'levels.2.blocks.26.norm1.weight',
    'levels.2.blocks.26.norm1.0.bias': 'levels.2.blocks.26.norm1.bias',
    'levels.2.blocks.26.norm2.0.weight': 'levels.2.blocks.26.norm2.weight',
    'levels.2.blocks.26.norm2.0.bias': 'levels.2.blocks.26.norm2.bias',
    'levels.2.blocks.27.norm1.0.weight': 'levels.2.blocks.27.norm1.weight',
    'levels.2.blocks.27.norm1.0.bias': 'levels.2.blocks.27.norm1.bias',
    'levels.2.blocks.27.norm2.0.weight': 'levels.2.blocks.27.norm2.weight',
    'levels.2.blocks.27.norm2.0.bias': 'levels.2.blocks.27.norm2.bias',
    'levels.2.blocks.28.norm1.0.weight': 'levels.2.blocks.28.norm1.weight',
    'levels.2.blocks.28.norm1.0.bias': 'levels.2.blocks.28.norm1.bias',
    'levels.2.blocks.28.norm2.0.weight': 'levels.2.blocks.28.norm2.weight',
    'levels.2.blocks.28.norm2.0.bias': 'levels.2.blocks.28.norm2.bias',
    'levels.2.blocks.29.norm1.0.weight': 'levels.2.blocks.29.norm1.weight',
    'levels.2.blocks.29.norm1.0.bias': 'levels.2.blocks.29.norm1.bias',
    'levels.2.blocks.29.norm2.0.weight': 'levels.2.blocks.29.norm2.weight',
    'levels.2.blocks.29.norm2.0.bias': 'levels.2.blocks.29.norm2.bias',
    'levels.2.blocks.30.norm1.0.weight': 'levels.2.blocks.30.norm1.weight',
    'levels.2.blocks.30.norm1.0.bias': 'levels.2.blocks.30.norm1.bias',
    'levels.2.blocks.30.norm2.0.weight': 'levels.2.blocks.30.norm2.weight',
    'levels.2.blocks.30.norm2.0.bias': 'levels.2.blocks.30.norm2.bias',
    'levels.2.blocks.31.norm1.0.weight': 'levels.2.blocks.31.norm1.weight',
    'levels.2.blocks.31.norm1.0.bias': 'levels.2.blocks.31.norm1.bias',
    'levels.2.blocks.31.norm2.0.weight': 'levels.2.blocks.31.norm2.weight',
    'levels.2.blocks.31.norm2.0.bias': 'levels.2.blocks.31.norm2.bias',
    'levels.2.blocks.32.norm1.0.weight': 'levels.2.blocks.32.norm1.weight',
    'levels.2.blocks.32.norm1.0.bias': 'levels.2.blocks.32.norm1.bias',
    'levels.2.blocks.32.norm2.0.weight': 'levels.2.blocks.32.norm2.weight',
    'levels.2.blocks.32.norm2.0.bias': 'levels.2.blocks.32.norm2.bias',
    'levels.2.blocks.33.norm1.0.weight': 'levels.2.blocks.33.norm1.weight',
    'levels.2.blocks.33.norm1.0.bias': 'levels.2.blocks.33.norm1.bias',
    'levels.2.blocks.33.norm2.0.weight': 'levels.2.blocks.33.norm2.weight',
    'levels.2.blocks.33.norm2.0.bias': 'levels.2.blocks.33.norm2.bias',
    'levels.2.blocks.34.norm1.0.weight': 'levels.2.blocks.34.norm1.weight',
    'levels.2.blocks.34.norm1.0.bias': 'levels.2.blocks.34.norm1.bias',
    'levels.2.blocks.34.norm2.0.weight': 'levels.2.blocks.34.norm2.weight',
    'levels.2.blocks.34.norm2.0.bias': 'levels.2.blocks.34.norm2.bias',
    'levels.2.blocks.35.norm1.0.weight': 'levels.2.blocks.35.norm1.weight',
    'levels.2.blocks.35.norm1.0.bias': 'levels.2.blocks.35.norm1.bias',
    'levels.2.blocks.35.norm2.0.weight': 'levels.2.blocks.35.norm2.weight',
    'levels.2.blocks.35.norm2.0.bias': 'levels.2.blocks.35.norm2.bias',
    'levels.2.blocks.36.norm1.0.weight': 'levels.2.blocks.36.norm1.weight',
    'levels.2.blocks.36.norm1.0.bias': 'levels.2.blocks.36.norm1.bias',
    'levels.2.blocks.36.norm2.0.weight': 'levels.2.blocks.36.norm2.weight',
    'levels.2.blocks.36.norm2.0.bias': 'levels.2.blocks.36.norm2.bias',
    'levels.2.blocks.37.norm1.0.weight': 'levels.2.blocks.37.norm1.weight',
    'levels.2.blocks.37.norm1.0.bias': 'levels.2.blocks.37.norm1.bias',
    'levels.2.blocks.37.norm2.0.weight': 'levels.2.blocks.37.norm2.weight',
    'levels.2.blocks.37.norm2.0.bias': 'levels.2.blocks.37.norm2.bias',
    'levels.2.blocks.38.norm1.0.weight': 'levels.2.blocks.38.norm1.weight',
    'levels.2.blocks.38.norm1.0.bias': 'levels.2.blocks.38.norm1.bias',
    'levels.2.blocks.38.norm2.0.weight': 'levels.2.blocks.38.norm2.weight',
    'levels.2.blocks.38.norm2.0.bias': 'levels.2.blocks.38.norm2.bias',
    'levels.2.blocks.39.norm1.0.weight': 'levels.2.blocks.39.norm1.weight',
    'levels.2.blocks.39.norm1.0.bias': 'levels.2.blocks.39.norm1.bias',
    'levels.2.blocks.39.norm2.0.weight': 'levels.2.blocks.39.norm2.weight',
    'levels.2.blocks.39.norm2.0.bias': 'levels.2.blocks.39.norm2.bias',
    'levels.2.blocks.40.norm1.0.weight': 'levels.2.blocks.40.norm1.weight',
    'levels.2.blocks.40.norm1.0.bias': 'levels.2.blocks.40.norm1.bias',
    'levels.2.blocks.40.norm2.0.weight': 'levels.2.blocks.40.norm2.weight',
    'levels.2.blocks.40.norm2.0.bias': 'levels.2.blocks.40.norm2.bias',
    'levels.2.blocks.41.norm1.0.weight': 'levels.2.blocks.41.norm1.weight',
    'levels.2.blocks.41.norm1.0.bias': 'levels.2.blocks.41.norm1.bias',
    'levels.2.blocks.41.norm2.0.weight': 'levels.2.blocks.41.norm2.weight',
    'levels.2.blocks.41.norm2.0.bias': 'levels.2.blocks.41.norm2.bias',
    'levels.2.blocks.42.norm1.0.weight': 'levels.2.blocks.42.norm1.weight',
    'levels.2.blocks.42.norm1.0.bias': 'levels.2.blocks.42.norm1.bias',
    'levels.2.blocks.42.norm2.0.weight': 'levels.2.blocks.42.norm2.weight',
    'levels.2.blocks.42.norm2.0.bias': 'levels.2.blocks.42.norm2.bias',
    'levels.2.blocks.43.norm1.0.weight': 'levels.2.blocks.43.norm1.weight',
    'levels.2.blocks.43.norm1.0.bias': 'levels.2.blocks.43.norm1.bias',
    'levels.2.blocks.43.norm2.0.weight': 'levels.2.blocks.43.norm2.weight',
    'levels.2.blocks.43.norm2.0.bias': 'levels.2.blocks.43.norm2.bias',
    'levels.2.blocks.44.norm1.0.weight': 'levels.2.blocks.44.norm1.weight',
    'levels.2.blocks.44.norm1.0.bias': 'levels.2.blocks.44.norm1.bias',
    'levels.2.blocks.44.norm2.0.weight': 'levels.2.blocks.44.norm2.weight',
    'levels.2.blocks.44.norm2.0.bias': 'levels.2.blocks.44.norm2.bias',
    'levels.2.blocks.45.norm1.0.weight': 'levels.2.blocks.45.norm1.weight',
    'levels.2.blocks.45.norm1.0.bias': 'levels.2.blocks.45.norm1.bias',
    'levels.2.blocks.45.norm2.0.weight': 'levels.2.blocks.45.norm2.weight',
    'levels.2.blocks.45.norm2.0.bias': 'levels.2.blocks.45.norm2.bias',
    'levels.2.blocks.46.norm1.0.weight': 'levels.2.blocks.46.norm1.weight',
    'levels.2.blocks.46.norm1.0.bias': 'levels.2.blocks.46.norm1.bias',
    'levels.2.blocks.46.norm2.0.weight': 'levels.2.blocks.46.norm2.weight',
    'levels.2.blocks.46.norm2.0.bias': 'levels.2.blocks.46.norm2.bias',
    'levels.2.blocks.47.norm1.0.weight': 'levels.2.blocks.47.norm1.weight',
    'levels.2.blocks.47.norm1.0.bias': 'levels.2.blocks.47.norm1.bias',
    'levels.2.blocks.47.norm2.0.weight': 'levels.2.blocks.47.norm2.weight',
    'levels.2.blocks.47.norm2.0.bias': 'levels.2.blocks.47.norm2.bias',
    'levels.2.norm.0.weight': 'levels.2.norm.weight',
    'levels.2.norm.0.bias': 'levels.2.norm.bias',
    'levels.2.downsample.norm.1.weight': 'levels.2.downsample.norm.weight',
    'levels.2.downsample.norm.1.bias': 'levels.2.downsample.norm.bias',
    'levels.3.blocks.0.norm1.0.weight': 'levels.3.blocks.0.norm1.weight',
    'levels.3.blocks.0.norm1.0.bias': 'levels.3.blocks.0.norm1.bias',
    'levels.3.blocks.0.norm2.0.weight': 'levels.3.blocks.0.norm2.weight',
    'levels.3.blocks.0.norm2.0.bias': 'levels.3.blocks.0.norm2.bias',
    'levels.3.blocks.1.norm1.0.weight': 'levels.3.blocks.1.norm1.weight',
    'levels.3.blocks.1.norm1.0.bias': 'levels.3.blocks.1.norm1.bias',
    'levels.3.blocks.1.norm2.0.weight': 'levels.3.blocks.1.norm2.weight',
    'levels.3.blocks.1.norm2.0.bias': 'levels.3.blocks.1.norm2.bias',
    'levels.3.blocks.2.norm1.0.weight': 'levels.3.blocks.2.norm1.weight',
    'levels.3.blocks.2.norm1.0.bias': 'levels.3.blocks.2.norm1.bias',
    'levels.3.blocks.2.norm2.0.weight': 'levels.3.blocks.2.norm2.weight',
    'levels.3.blocks.2.norm2.0.bias': 'levels.3.blocks.2.norm2.bias',
    'levels.3.blocks.3.norm1.0.weight': 'levels.3.blocks.3.norm1.weight',
    'levels.3.blocks.3.norm1.0.bias': 'levels.3.blocks.3.norm1.bias',
    'levels.3.blocks.3.norm2.0.weight': 'levels.3.blocks.3.norm2.weight',
    'levels.3.blocks.3.norm2.0.bias': 'levels.3.blocks.3.norm2.bias',
    'levels.3.norm.0.weight': 'levels.3.norm.weight',
    'levels.3.norm.0.bias': 'levels.3.norm.bias',
    'clip_projector.norm1_q.0.weight': 'clip_projector.q_norm1.weight',
    'clip_projector.norm1_q.0.bias': 'clip_projector.q_norm1.bias',
    'clip_projector.norm1_k.0.weight': 'clip_projector.k_norm1.weight',
    'clip_projector.norm1_k.0.bias': 'clip_projector.k_norm1.bias',
    'clip_projector.norm1_v.0.weight': 'clip_projector.v_norm1.weight',
    'clip_projector.norm1_v.0.bias': 'clip_projector.v_norm1.bias',
    'clip_projector.cross_dcn.q_bias': 'clip_projector.cross_dcn.query_bias',
    'clip_projector.cross_dcn.k_bias': 'clip_projector.cross_dcn.key_bias',
    'clip_projector.cross_dcn.v_bias': 'clip_projector.cross_dcn.value_bias',
    'clip_projector.cross_dcn.q.weight': 'clip_projector.cross_dcn.query.weight', 
    'clip_projector.cross_dcn.k.weight': 'clip_projector.cross_dcn.key.weight', 
    'clip_projector.cross_dcn.v.weight': 'clip_projector.cross_dcn.value.weight',
    'clip_projector.cross_dcn.proj.weight': 'clip_projector.cross_dcn.projection.weight', 
    'clip_projector.cross_dcn.proj.bias': 'clip_projector.cross_dcn.projection.bias',
    'fc_norm.0.weight' : 'fc_norm.weight', 
    'fc_norm.0.bias' : 'fc_norm.bias'
}