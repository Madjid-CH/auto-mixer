from auto_mixer.modules.base import MulticlassMixer, MultilabelMixer


class MultilabelImageMixer(MultilabelMixer):

    def get_logits(self, images):
        logits = self.backbone(images)
        logits = self.classifier(logits)
        logits = logits.mean(dim=1)
        return logits


class MulticlassImageMixer(MulticlassMixer):

    def get_logits(self, images):
        logits = self.backbone(images)
        logits = self.classifier(logits)
        logits = logits.mean(dim=1)
        return logits
