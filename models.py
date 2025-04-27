from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module): #best was lr = 1e-2 and epoch 50
    class Block(torch.nn.Module):
        def __init__(self,in_channels,out_channels,stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size-1) // 2
            self.conv = torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
            self.relu = torch.nn.ReLU()
            
        def forward (self,x):
            return self.relu(self.conv(x))
        
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        layers = [16,32,64],
        kernel_size = 3,
        stride = 1
    ):
        #A convolutional network for image classification.

        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        cnn_layers = [
            torch.nn.Conv2d(in_channels,64,kernel_size=11,stride=2,padding=5),
            torch.nn.ReLU()
        ]
        c1 = 64
        for _ in range(3):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1,c2,stride=1))
            c1 = c2
        cnn_layers.append(torch.nn.Conv2d(c1,num_classes,kernel_size=1))
        cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1))
        self.network = torch.nn.Sequential(*cnn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        return x.view(x.size(0),-1)


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).argmax(dim=1)

MODEL_FACTORY = {
    "classifier": Classifier,
}

def load_model(
    model_name: str,
    with_weights: bool = False,
    model_path: str = None,
    **model_kwargs,
) -> torch.nn.Module:
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        if model_path is None:
            model_path = DIR / f"{model_name}.th"
        else:
            model_path = Path(model_path)

        print(f"Looking for model at: {model_path}")
        assert model_path.exists(), f"{model_path} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(f"Failed to load {model_path.name}") from e

    return m

def save_model(model: torch.nn.Module) -> str:
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) == m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path

def calculate_model_size_mb(model: torch.nn.Module) -> float:

    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024

def debug_model(batch_size: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
