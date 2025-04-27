import torch

class AccuracyMetric:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        """
        should be called before each epoch
        """
        self.correct = 0
        self.total = 0

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Updates using predictions and ground truth labels

        Args:
            preds (torch.LongTensor): (b,) or (b, h, w) tensor with class predictions
            labels (torch.LongTensor): (b,) or (b, h, w) tensor with ground truth class labels
        """
        self.correct += (preds.type_as(labels) == labels).sum().item()
        self.total += labels.numel()

    def compute(self) -> dict[str, float]:
        return {
            "accuracy": self.correct / (self.total + 1e-5),
            "num_samples": self.total,
        }


