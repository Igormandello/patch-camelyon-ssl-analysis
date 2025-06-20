import torch
from lightning.pytorch.callbacks import Callback

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from dataset_pcam import PCamDataModule

class SGDBenchmark(Callback):
    def __init__(self, datamodule: PCamDataModule, backbone: torch.nn.Module, train_samples: int, predictor_epochs: int = 0, use_test_set: bool = False) -> None:
        super().__init__()

        self.train_dataloader = datamodule.train_dataloader(samples_per_class=train_samples, shuffle=False)
        if use_test_set:
            self.test_dataloader = datamodule.test_dataloader()
        else:
            self.test_dataloader = datamodule.val_dataloader()

        self.backbone = backbone
        self.predictor_epochs = predictor_epochs

    def get_accuracy(self):
        self.sgd = SGDClassifier(random_state=42)
        backbone_device = next(self.backbone.parameters()).device

        self.backbone.eval()
        self.__fit_sgd(backbone_device)

        test_y = []
        y_pred = []
        with torch.no_grad():
            for (X, y) in self.test_dataloader:
                X = self.backbone(X.to(backbone_device)).flatten(start_dim=1)
                test_y.extend(y.cpu())
                y_pred.extend(self.sgd.predict(X.cpu()))

        return accuracy_score(test_y, y_pred)

    def on_train_epoch_end(self, trainer, model):
        if (self.predictor_epochs <= 0 or trainer.current_epoch % (self.predictor_epochs + 1) != 0):
            return

        acc = self.get_accuracy()
        model.log("SGD_acc", acc, on_step=False, on_epoch=True, sync_dist=True)

    def __fit_sgd(self, device):
        with torch.no_grad():
            for (X, y) in self.train_dataloader:
                X = self.backbone(X.to(device)).flatten(start_dim=1)
                self.sgd.partial_fit(X.cpu(), y.cpu(), classes=[0, 1])
                break