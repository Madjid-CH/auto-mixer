
from auto_mixer.modules.base import MultilabelMixer, MulticlassMixer


class MultilabelTextMixer(MultilabelMixer):
    def get_logits(self, embeddings):
        logits = self.backbone(embeddings)
        logits = logits.mean(dim=-1).squeeze(-1)  # TODO check if this is correct
        logits = self.classifier(logits)
        return logits


class MulticlassTextMixer(MulticlassMixer):

    def get_logits(self, embeddings):
        logits = self.backbone(embeddings)
        logits = logits.mean(dim=-1).squeeze(-1)
        logits = self.classifier(logits)
        return logits
