import torch
from omegaconf import OmegaConf

from models import MultilabelImageMixer, MultilabelTextMixer
from models.multimodal_models import MultiLabelMultiLoss


def test_forward_pass():
    cfgs_path = "C:/Users/M/PycharmProjects/auto-mixer/auto_mixer/cfg/image_models/hypermixer_s.yml"
    cfgs = OmegaConf.load(cfgs_path)
    image_size = (256, 256)
    train_cfg = OmegaConf.load("C:/Users/M/PycharmProjects/auto-mixer/auto_mixer/cfg/micro_train.yml")
    image_mixer = (
        cfgs.block_type, MultilabelImageMixer(image_size=image_size,
                                              target_length=23,
                                              model_cfg=cfgs,
                                              optimizer_cfg=train_cfg.optimizer
                                              ).backbone
    )
    image_mixer[1].classifier_input_dim = 640
    cfgs_path = "C:/Users/M/PycharmProjects/auto-mixer/auto_mixer/cfg/text_models/monarchmixer_s.yml"
    cfgs = OmegaConf.load(cfgs_path)
    hidden_dim = 768
    patch_size = 512
    text_mixer = (
        cfgs.block_type, MultilabelTextMixer(
            hidden_dim=hidden_dim, patch_size=patch_size,
            target_length=23, model_cfg=cfgs,
            optimizer_cfg=train_cfg.optimizer
        ).backbone
    )
    text_mixer[1].classifier_input_dim = 729
    encoders = {"images": image_mixer, "texts": text_mixer}
    cfgs_path = "C:/Users/M/PycharmProjects/auto-mixer/auto_mixer/cfg/fusion_models/hypermixer.yml"
    cfgs = OmegaConf.load(cfgs_path)
    model = MultiLabelMultiLoss(
        encoders=encoders,
        target_length=23,
        model_cfg=cfgs,
        optimizer_cfg=train_cfg.optimizer
    ).cpu()

    batch = {"images": torch.rand(10, 3, 256, 256), "texts": torch.rand(10, 512, 768), "labels": torch.rand(10, 23)}
    with torch.no_grad():
        output = model.shared_step(batch)
    assert output['preds'].shape == (10, 23)
