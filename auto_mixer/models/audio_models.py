from omegaconf import DictConfig

from auto_mixer import modules
from auto_mixer.models.base import MulticlassMixer, MultilabelMixer
from auto_mixer.modules import TwoLayeredPerceptron


class MultilabelAudioMixer(MultilabelMixer):
    def __init__(self, image_size, target_length, model_cfg: DictConfig, optimizer_cfg: DictConfig,
                 **kwargs):
        super(MultilabelMixer, self).__init__(target_length=target_length, optimizer_cfg=optimizer_cfg, **kwargs)
        model_cls = getattr(modules, model_cfg.block_type)
        self.backbone = model_cls(image_size=image_size, **model_cfg.image, dropout=model_cfg.dropout)
        classifier_input_dim = model_cfg.get('classifier_input_dim', 301)
        self.backbone.classifier_input_dim = classifier_input_dim
        self.classifier = TwoLayeredPerceptron(input_dim=classifier_input_dim, hidden_dim=512,
                                               output_dim=self.target_length)

    def get_logits(self, embeddings):
        logits = self.backbone(embeddings)
        logits = logits.mean(dim=-1).squeeze(-1)  # TODO check if this is correct
        logits = self.classifier(logits)
        return logits

    def shared_step(self, batch, **kwargs):
        labels = batch['labels']
        texts = batch['audio']
        return self.predict(texts, labels)


class MulticlassAudioMixer(MulticlassMixer):

    def __init__(self, image_size, target_length, model_cfg: DictConfig, optimizer_cfg: DictConfig,
                 **kwargs):
        super(MulticlassMixer, self).__init__(target_length=target_length, optimizer_cfg=optimizer_cfg, **kwargs)
        model_cls = getattr(modules, model_cfg.block_type)
        self.backbone = model_cls(image_size=image_size, **model_cfg.audio,
                                  dropout=model_cfg.dropout)
        classifier_input_dim = model_cfg.classifier_input_dim if hasattr(model_cfg, 'classifier_input_dim') else 512
        self.backbone.classifier_input_dim = classifier_input_dim
        self.classifier = TwoLayeredPerceptron(input_dim=classifier_input_dim, hidden_dim=512,
                                               output_dim=self.target_length)

    def get_logits(self, embeddings):
        logits = self.backbone(embeddings)
        logits = logits.mean(dim=1)
        logits = self.classifier(logits)
        return logits

    def shared_step(self, batch, **kwargs):
        labels = batch['labels']
        texts = batch['audio']
        return self.predict(texts, labels)
