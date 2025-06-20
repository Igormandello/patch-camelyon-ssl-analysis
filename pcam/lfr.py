import torch
from minerva.models.ssl.lfr import LearnFromRandomnessModel

class LFRModel(LearnFromRandomnessModel):
    def __init__(self, backbone: torch.nn.Module, projectors: list[torch.nn.Module], predictors: torch.nn.ModuleList, predictor_epochs: int, lr: float):
        super().__init__(
            backbone=backbone, 
            projectors=torch.nn.ModuleList(projectors),
            predictors=predictors,
            predictor_training_epochs=predictor_epochs,
            learning_rate=lr
        )

        self.lr = lr
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        backbone_opt, predictors_opt = self.optimizers()
        loss = self._single_step(batch, batch_idx, "train")

        backbone_opt.zero_grad()
        predictors_opt.zero_grad()
        self.manual_backward(loss)
        backbone_opt.step()
        predictors_opt.step()

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def configure_optimizers(self):
        super().configure_optimizers()
        backbone_optimizer = torch.optim.SGD(self.backbone.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        predictors_optimizer = torch.optim.SGD(self.predictors.parameters(), lr=self.lr, momentum=0.9)
        # optimizer = torch.optim.Adam(
        #     self.parameters(),
        #     lr=self.lr,
        #     weight_decay=3e-4,
        #     betas=(0.9, 0.999),
        # )

        return (
            {
                "optimizer": backbone_optimizer,
                "lr_scheduler": {
                    # "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.001, factor=0.1),
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(backbone_optimizer, T_max=20, eta_min=1e-6),
                    "interval": "epoch",
                    "frequency": 1,
                    # "monitor": "val_loss",
                    # "strict": True
                }
            },
            {
                "optimizer": predictors_optimizer
            }
        )