from abc import ABC, abstractmethod
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as ttf
from tools.model_saver import StoredModel, ModelSaver
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer(ABC):
    def __init__(self, 
                    model_id: str, 
                    model: nn.Module, 
                    optimizer: optim.Optimizer, 
                    criterion, 
                    scheduler, 
                    device, 
                    checkpoints: str, 
                    max_epochs: int=10, 
                    epoch_start: int=0, 
                    log_checkpoints:str=None, 
                    model_saver_mode='min') -> None:

        self.model_id = model_id
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = max_epochs
        self.epoch_start = epoch_start
        self.epochs = epoch_start
        self.checkpoints = checkpoints
        if log_checkpoints:
            self.log_checkpoints = log_checkpoints
        else:
            self.log_checkpoints = checkpoints
        
        if not os.path.exists(self.checkpoints):
            os.mkdir(self.checkpoints)
        if not os.path.exists(self.log_checkpoints):
            os.mkdir(self.log_checkpoints)
        
        self.create_model_saver(checkpoints, mode=model_saver_mode)
        
        pass
    
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def _train(self):
        pass
    
    @abstractmethod
    def _val(self):
        pass
    
    @abstractmethod
    def inference(self):
        pass
    
    def create_model_saver(self, checkpoints, mode='min'):
        self.model_saver = ModelSaver(mode=mode, model_id=self.model_id, checkpoints=checkpoints)
        self.model_saver.save_spec(str(self.model))
        self.model_saver.save_spec(str(self.optimizer))
        self.model_saver.save_spec(str(self.criterion))
        self.model_saver.save_spec(str(self.scheduler))
    
    def save_log(self, log: str):
        with open(f"{self.log_checkpoints}/{self.model_id}/log.txt", mode='a') as f:
            f.write(f"{log}\n")

# class Template(Trainer):
#     def __init__(self, 
#                     model_id: str, 
#                     model: nn.Module, 
#                     optimizer: optim.Optimizer, 
#                     criterion, 
#                     scheduler, 
#                     device, 
#                     checkpoints: str, 
#                     max_epochs: int=10, 
#                     epoch_start: int=0, 
#                     log_checkpoints: str=None) -> None:
#         super().__init__(model_id, model, optimizer, criterion, scheduler, device, checkpoints, max_epochs, epoch_start, log_checkpoints)

class ClassifierTrainer(Trainer):
    def __init__(self, 
                    model_id: str, 
                    model: nn.Module, 
                    optimizer: optim.Optimizer, 
                    criterion, 
                    scheduler, 
                    device, 
                    checkpoints: str, 
                    max_epochs: int=10, 
                    epoch_start: int=0, 
                    log_checkpoints: str=None, 
                    scaler=None, 
                    model_saver_mode='max',
                    batch_size=64) -> None:
        super().__init__(model_id, model, optimizer, criterion, scheduler, device, checkpoints, max_epochs, epoch_start, log_checkpoints, model_saver_mode=model_saver_mode)

        self.scaler = scaler
        self.batch_size = batch_size
        self.train_losses = []
        self.val_losses = []

    def _train(self, train_loader, batch_bar=None):
        self.model.train()
        start_time = time.time()
        running_loss = 0.0
        torch.cuda.empty_cache()

        num_correct = 0
        running_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            self.optimizer.zero_grad()
            x, y = x.float().to(self.device), y.to(self.device)

            with torch.cuda.amp.autocast():     
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
            
            num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
            running_loss += float(loss)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # self.scheduler.step()

            if batch_bar:
                batch_bar.set_postfix(
                    acc="{:.04f}%".format(100 * num_correct / ((i + 1) * self.batch_size)),
                    loss="{:.04f}".format(float(running_loss / ((i + 1)))),
                    num_correct=num_correct,
                    lr="{:.04f}".format(float(self.optimizer.param_groups[0]['lr'])))
                batch_bar.update()
            
            del x
            del y
            torch.cuda.empty_cache()
        if batch_bar:
            batch_bar.close()
        epoch_time = time.time() - start_time

        train_acc = 100 * num_correct / (len(train_loader) * self.batch_size)
        train_loss = running_loss / len(train_loader)
        self.epochs += 1
        train_log = f"Train Epoch {self.epochs}/{self.epoch_start+self.max_epochs}: train_acc={train_acc}% | loss={train_loss} | lr={float(self.optimizer.param_groups[0]['lr'])} | ({int(epoch_time//60)}min{int(epoch_time%60)}s)"

        self.train_losses.append(train_loss)
        self.save_log(train_log)

        return train_loss
    
    def _val(self, val_loader, batch_bar=None):
        self.model.eval()
        start_time = time.time()
        torch.cuda.empty_cache()
        running_val_loss = 0.0
        num_correct = 0

        for i, (x, y) in enumerate(val_loader):
            x, y = x.float().to(self.device), y.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
            
            num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
            running_val_loss += float(loss)
            if batch_bar:
                batch_bar.set_postfix(acc="{:.04f}%".format(100 * num_correct / ((i + 1) * self.batch_size)))
                
                batch_bar.update()
        if batch_bar:
            batch_bar.close()
        epoch_time = time.time() - start_time
        val_acc = 100 * num_correct / (len(val_loader) * self.batch_size)
        val_loss = running_val_loss / len(val_loader)
        val_log = f" - Val Epoch {self.epochs}/{self.epoch_start+self.max_epochs}: val_acc={val_acc}% | loss={val_loss} | ({int(epoch_time//60)}min{int(epoch_time%60)}s)"

        self.val_losses.append(val_loss)
        self.save_log(val_log)

        return val_acc, val_loss

    def fit(self, train_loader, val_loader):
        total_epochs = self.epoch_start + self.max_epochs
        
        for epoch in range(self.epoch_start, total_epochs):

            batch_bar_train = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc=f"Train Epoch {epoch+1}/{total_epochs}:")
            train_loss = self._train(train_loader, batch_bar_train)

            batch_bar_val = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc=f"Val Epoch {epoch+1}:")
            val_acc, val_loss = self._val(val_loader, batch_bar_val)
            
            self.scheduler.step(val_loss)
            
            stats = {
                "epoch": epoch+1,
                "lr": self.optimizer.param_groups[0]['lr'], 
                "train_loss": train_loss, 
                "val_loss": val_loss, 
            }
            self.model_saver.save(StoredModel(self.model, self.optimizer, self.scheduler, self.criterion), stats, val_acc)

        print(f"The best epoch is: {self.model_saver.best_epoch}")
        with open(f"{self.log_checkpoints}/{self.model_id}/log.txt", mode='a') as f:
            f.write(f"The best epoch is: {self.model_saver.best_epoch}\n")

        plt.plot(range(self.epoch_start+1, total_epochs+1), self.train_losses, label = "train loss")
        plt.plot(range(self.epoch_start+1, total_epochs+1), self.val_losses, label = "val loss")
        plt.legend()
        plt.xticks(range(self.epoch_start+1, total_epochs+1))
        plt.xlabel("# of Epochs")
        plt.ylabel("Loss")
        plt.savefig(f"{self.checkpoints}/{self.model_id}/loss_curves_{self.model_id}.png")
        plt.show()

    def inference(self):
        TEST_DIR = "..."
        transforms = ttf.Compose([ttf.Resize((224, 224)), ttf.ToTensor()])
        test_dataset = torchvision.datasets.ImageFolder(TEST_DIR, transform=transforms)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                    drop_last=False, num_workers=1)
        
        self.model.eval()
        num_correct = 0
        batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')
        # res = []
        for i, (x, y) in enumerate(test_loader):

            # TODO: Finish predicting on the test set.
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                outputs = self.model(x)
            
            num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
            # res.extend(torch.argmax(outputs, axis=1).tolist())

            batch_bar.update()
            
        batch_bar.close()
        
        return num_correct / len(test_dataset)
