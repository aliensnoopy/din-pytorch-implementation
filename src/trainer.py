import torch.cuda
import torch.nn.functional as F

from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.collator import Collator
from src.config import DinConfig
from src.dataset import AmazonDataset
from src.model import DeepInterestNetwork


@dataclass
class TrainingArguments:
    model_config: DinConfig
    train_data_path: str
    test_data_path: str
    max_len: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    logging_steps: int
    evaluate_steps: int


class Trainer:
    def __init__(self, args: TrainingArguments):
        super().__init__()
        self.args = args
        device_count = torch.cuda.device_count()
        print(f"There are {device_count} GPUs.")

        collator = Collator(max_len=args.max_len)
        train_dataset = AmazonDataset(data_file=args.train_data_path)
        self.train_dataloader = DataLoader(dataset=train_dataset,
                                           batch_size=max(1, device_count) * args.batch_size,
                                           collate_fn=collator,
                                           num_workers=8,
                                           shuffle=True,
                                           pin_memory=True)

        test_dataset = AmazonDataset(data_file=args.test_data_path)
        self.test_dataloader = DataLoader(dataset=test_dataset,
                                          batch_size=max(1, device_count) * args.batch_size,
                                          collate_fn=collator,
                                          num_workers=8,
                                          shuffle=False,
                                          pin_memory=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, self.optimizer = self._init_model_and_optimizer(device_count)

        self.writer = SummaryWriter(log_dir="logs")

        print(f"Totally {len(train_dataset)} / {len(test_dataset)} (train / test) samples, "
              f"{len(self.train_dataloader)} / {len(self.test_dataloader)} (train / test) batches.")

    def _init_model_and_optimizer(self, device_count: int):
        model = DeepInterestNetwork(config=self.args.model_config)

        def weights_init(m):
            if isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight.data, -1, 1)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.args.learning_rate,
                                     betas=(self.args.adam_beta1, self.args.adam_beta2))
        if device_count > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        model.apply(weights_init)

        return model, optimizer

    def train(self):
        global_step = 1
        for epoch_idx in range(self.args.num_epochs):
            for batch_idx, batch_inputs in enumerate(self.train_dataloader, start=1):
                self.model.train()
                inputs = {k: v.to(self.device) for k, v in batch_inputs.items() if k != "label"}
                labels = batch_inputs["label"].to(self.device)
                self.model.zero_grad()
                output_logits = self.model(**inputs)
                loss = F.cross_entropy(output_logits, labels, reduction="mean")
                loss.backward()
                self.optimizer.step()

                pred = torch.argmax(output_logits, dim=-1)
                accuracy = (pred == labels).float().mean().item()
                auc = roc_auc_score(labels.detach().cpu(), pred.detach().cpu())

                global_step += 1

                if global_step % self.args.logging_steps == 0:
                    print(f"{'[Train]':<8} global step: {global_step:>4}, loss: {loss.item():.5f}, "
                          f"accuracy: {accuracy:.5f}, auc: {auc:.5f}")
                    self.writer.add_scalar("loss/train", loss.item(), global_step)
                    self.writer.add_scalar("accuracy/train", accuracy, global_step)
                    self.writer.add_scalar("auc/train", auc, global_step)

                if global_step % self.args.evaluate_steps == 0:
                    self.evaluate(global_step)

    def evaluate(self, global_step: int):
        self.model.eval()

        total_loss = 0.0
        predictions, references = [], []
        for batch_idx, batch_inputs in enumerate(self.test_dataloader, start=1):
            inputs = {k: v.to(self.device) for k, v in batch_inputs.items() if k != "label"}
            labels = batch_inputs["label"].to(self.device)
            with torch.no_grad():
                output_logits = self.model(**inputs)
                loss = F.cross_entropy(output_logits, labels, reduction="sum")
                total_loss += loss.item()
                pred = torch.argmax(output_logits, dim=-1)
                predictions.append(pred.detach().cpu())
                references.append(labels.detach().cpu())

        predictions = torch.cat(predictions, dim=0)
        references = torch.cat(references, dim=0)

        mean_loss = total_loss / predictions.size(0)
        accuracy = (predictions == references).float().mean().item()
        auc = roc_auc_score(references, predictions)

        print(f"[Evaluate] global step: {global_step:>4}, loss: {mean_loss:.5f}, "
              f"accuracy: {accuracy:.5f}, auc: {auc:.5f}")
        self.writer.add_scalar("loss/test", mean_loss, global_step)
        self.writer.add_scalar("accuracy/test", accuracy, global_step)
        self.writer.add_scalar("auc/test", auc, global_step)

