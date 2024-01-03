from easydict import EasyDict as EDict

from Models.gmmformer.model import GMMFormer_Net

def get_models(cfg):
    model_config = EDict(
        visual_input_size=cfg['visual_feat_dim'],
        query_input_size=cfg['q_feat_size'],
        hidden_size=cfg['hidden_size'],  # hidden dimension
        max_ctx_l=cfg['max_ctx_l'],
        max_desc_l=cfg['max_desc_l'],
        map_size=cfg['map_size'],
        input_drop=cfg['input_drop'],
        drop=cfg['drop'],
        n_heads=cfg['n_heads'],  # self-att heads
        initializer_range=cfg['initializer_range'],  # for linear layer
        margin=cfg['margin'],  # margin for ranking loss
        use_hard_negative=False,  # reset at each epoch
        hard_pool_size=cfg['hard_pool_size'])
    model = GMMFormer_Net(model_config)
    return model
