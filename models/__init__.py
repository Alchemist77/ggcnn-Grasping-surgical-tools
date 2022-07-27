def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GGCNN2
        return GGCNN2
    elif network_name == 'vae_ggcnn':
        from .vae_ggcnn import VAE_GGCNN
        return VAE_GGCNN
    elif network_name == 'vae_ggcnn2':
        from .vae_ggcnn2 import VAE_GGCNN2
        return VAE_GGCNN2
    elif network_name == 'vit_ggcnn':
        from .vit_ggcnn import VisionTransformer
        return VisionTransformer
    elif network_name == 'unet_ggcnn':
        from .unet_ggcnn import UNet
        return UNet
    elif network_name == 'auto_ggcnn':
        from .auto_ggcnn import AUTO_GGCNN
        return AUTO_GGCNN
    elif network_name == 'sparse_ggcnn':
        from .sparse_ggcnn import SPARSE_GGCNN
        return SPARSE_GGCNN
    elif network_name == 'sparse_ggcnn2':
        from .sparse_ggcnn2 import SPARSE_GGCNN2
        return SPARSE_GGCNN2
    elif network_name == 'contractive_ggcnn':
        from .contractive_ggcnn import CONTRACTIVE_GGCNN
        return CONTRACTIVE_GGCNN
    elif network_name == 'contractive_ggcnn2':
        from .contractive_ggcnn2 import CONTRACTIVE_GGCNN2
        return CONTRACTIVE_GGCNN2
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
