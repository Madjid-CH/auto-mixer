from omegaconf import DictConfig

from auto_mixer import modules
from auto_mixer.models.base import MulticlassMixer, MultilabelMixer
from auto_mixer.modules import TwoLayeredPerceptron


class MultilabelImageMixer(MultilabelMixer):

    def __init__(self, image_size, target_length: int, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(MultilabelMixer, self).__init__(target_length=target_length, optimizer_cfg=optimizer_cfg, **kwargs)
        model_cls = getattr(modules, model_cfg.block_type)
        self.backbone = model_cls(image_size=image_size, **model_cfg.image, dropout=model_cfg.dropout)
        classifier_input_dim = model_cfg.get('classifier_input_dim', 512)
        self.classifier = TwoLayeredPerceptron(input_dim=classifier_input_dim, hidden_dim=512,
                                               output_dim=self.target_length)

    def shared_step(self, batch, **kwargs):
        labels = batch['labels']
        images = batch['images']
        return self.predict(images, labels)

    def get_logits(self, images):
        logits = self.backbone(images)
        logits = self.classifier(logits)
        logits = logits.mean(dim=1)
        return logits


class MulticlassImageMixer(MulticlassMixer):
    def __init__(self, image_size, target_length, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(MulticlassMixer, self).__init__(target_length=target_length, optimizer_cfg=optimizer_cfg, **kwargs)
        model_cls = getattr(modules, model_cfg.block_type)
        self.backbone = model_cls(image_size=image_size, **model_cfg.image, dropout=model_cfg.dropout)
        self.classifier = TwoLayeredPerceptron(input_dim=512, hidden_dim=512, output_dim=self.target_length)

    def get_logits(self, images):
        logits = self.backbone(images)
        logits = self.classifier(logits)
        logits = logits.mean(dim=1)
        return logits

    def shared_step(self, batch, **kwargs):
        labels = batch['labels']
        images = batch['images']
        return self.predict(images, labels)
