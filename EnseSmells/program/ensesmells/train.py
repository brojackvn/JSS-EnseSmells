from tqdm import tqdm
import config
import gc
import os

# sklearn lib
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# pytorch lib
import torch
from torch.cuda.amp import GradScaler, autocast

# my tool
from utils import write_file
from config import Config, argument

args = argument()

class Trainer():
    def __init__(self, device, dataloader, model, loss_fns, optimizer, scheduler=None):
        self.device = device
        self.train_loader, self.valid_loader = dataloader
        self.model = model
        self.train_loss_fn, self.valid_loss_fn = loss_fns
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def train_one_epoch(self):
        '''
            Train the model for 1 epochs
        '''
        self.model.train()
        train_pbar = tqdm(enumerate(self.train_loader), total=(len(self.train_loader)))
        train_preds, train_targets = [], []
        
        for idx, cache in train_pbar:
            if args.model == 'DeepSmells_METRICS' or args.model == 'DeepSmells_TokenIndexing_METRICS':
                inputs = self._convert_if_not_tensor(cache[0], dtype=torch.float)
                inputs_metrics = self._convert_if_not_tensor(cache[1], dtype=torch.float)
                targets = self._convert_if_not_tensor(cache[2], dtype=torch.float)
            else: 
                inputs = self._convert_if_not_tensor(cache[0], dtype=torch.float)
                targets = self._convert_if_not_tensor(cache[1], dtype=torch.float)

            with autocast(enabled=True):
                if args.model == 'DeepSmells_METRICS' or args.model == 'DeepSmells_TokenIndexing_METRICS':
                    outputs = self.model(inputs, inputs_metrics)
                else:
                    outputs = self.model(inputs)

                loss = self.train_loss_fn(outputs, targets)
                loss_itm = loss.item()

                train_pbar.set_description(f'train_loss: {loss_itm:.4f}')

                config.Config.SCALER.scale(loss).backward()
                config.Config.SCALER.step(self.optimizer)
                config.Config.SCALER.update()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

            train_preds.extend(outputs.cpu().detach().numpy().tolist())
            train_targets.extend(targets.cpu().detach().numpy().tolist())
        
        del outputs, targets, inputs, loss_itm, loss
        gc.collect()

        return train_preds, train_targets

    @torch.no_grad()
    def valid_one_epoch(self):
        '''
            Validates the model for 1 epoch
        '''
        self.model.eval()
        valid_pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))
        valid_preds, valid_targets = [], []

        for idx, cache in valid_pbar:
            if args.model == 'DeepSmells_METRICS' or args.model == 'DeepSmells_TokenIndexing_METRICS':
                inputs = self._convert_if_not_tensor(cache[0], dtype=torch.float)
                inputs_metrics = self._convert_if_not_tensor(cache[1], dtype=torch.float)
                targets = self._convert_if_not_tensor(cache[2], dtype=torch.float)
            else: 
                inputs = self._convert_if_not_tensor(cache[0], dtype=torch.float)
                targets = self._convert_if_not_tensor(cache[1], dtype=torch.float)

            if args.model == 'DeepSmells_METRICS' or args.model == 'DeepSmells_TokenIndexing_METRICS':
                outputs = self.model(inputs, inputs_metrics)
            else:
                outputs = self.model(inputs)

            valid_loss = self.valid_loss_fn(outputs, targets)
            valid_pbar.set_description(desc=f"valid_loss: {valid_loss:.4f}")

            valid_preds.extend(outputs.cpu().detach().numpy().tolist())
            valid_targets.extend(targets.cpu().detach().numpy().tolist())
        
        del outputs, targets, inputs, valid_loss
        gc.collect()

        return valid_preds, valid_targets
            

    def fit(self,
            epochs: int = 70,
            checkpoint_dir: str = None,
            custom_name: str = f"model.pth",
            track_dir: str = f'./track/',
            threshold: float = 0.5,
            ):
        '''
            Low-effort alternative for doing the complete training and validation process
        '''
        best_precision = 0
        best_recall = 0
        best_f1 = 0
        best_auc = 0
        best_mcc = 0


        for epx in range(epochs):
            print(f"{'='*25} Epoch: {epx+1} / {epochs} {'='*25}")
            write_file(track_dir, f"{'='*25} Epoch: {epx+1} / {epochs} {'='*25}\n")

            # Result of epoch train
            train_preds, train_targets = self.train_one_epoch()
            train_loss_all = self.train_loss_fn(torch.Tensor(train_preds), torch.Tensor(train_targets))

            train_preds = [True if torch.sigmoid(torch.tensor(pred[0])) >= threshold else False for pred in train_preds]
            train_targets = [True if target[0] == 1.0 else False for target in train_targets]
            train_precision, train_recall, train_f1, train_auc, train_mcc = self.evaluation_metrics(train_preds, train_targets)
            print(f"Training Loss: {train_loss_all:.4f}")
            write_file(track_dir, f"Training Loss: {train_loss_all:.4f}\n")

            # Result of epoch valid
            valid_preds, valid_targets = self.valid_one_epoch()
            valid_loss_all = self.valid_loss_fn(torch.Tensor(valid_preds), torch.Tensor(valid_targets))

            valid_preds = [True if torch.sigmoid(torch.tensor(pred[0])) >= threshold else False for pred in valid_preds]
            valid_targets = [True if target[0] == 1.0 else False for target in valid_targets]
            valid_precision, valid_recall, valid_f1, valid_auc, valid_mcc = self.evaluation_metrics(valid_preds, valid_targets)
            print(f'Validation loss: {valid_loss_all:.4f}')
            write_file(track_dir, f"Validation loss: {valid_loss_all:.4f}\n")

            if valid_f1 > best_f1:
                best_f1 = valid_f1
                best_precision = valid_precision
                best_recall = valid_recall
                best_auc = valid_auc
                best_mcc = valid_mcc
                if checkpoint_dir is not None:
                    self.save_model(epx+1, checkpoint_dir, custom_name)
                
        print(f'Precision: {best_precision}')
        print(f'Recall   : {best_recall}')
        print(f'F1       : {best_f1}')
        print(f'AUC      : {best_auc}')
        print(f'MCC      : {best_mcc}')
        write_file(track_dir, f"{'='*65}\n")
        write_file(track_dir, f"Precision: {best_precision}\nRecall   : {best_recall}\nF1       : {best_f1}\nAUC      : {best_auc}\nMCC      : {best_mcc}\n")
        return best_precision, best_recall, best_f1, best_auc, best_mcc

    def save_model(self, epoch, path, name, verbose=False):
        '''
            Save  the model at the provided destination
        '''
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except:
            print('Errors encountered while making the output directory')
        
        state = {
            'epoch': epoch, 
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state, os.path.join(path,f"{name}.pth"))
        if verbose:
            print(f'Model saved at: {os.path.join(path, f"{name}.pth")}')
    
    def evaluation_metrics(self, preds, targets):
        '''
            Evaluation matrics: Precision, Recall, F1, MCC
        '''
        tn, fp, fn, tp = confusion_matrix(preds, targets).ravel()
        
        # Precision
        precision = tp/(tp+fp) 
        
        # Recall
        recall = tp/(tp+fn)

        # f1-score
        fscore = metrics.f1_score(preds, targets)

        # AUC
        fpr, tpr, thresholds = metrics.roc_curve(targets, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        # MCC
        mcc = metrics.matthews_corrcoef(preds, targets)

        return precision, recall, fscore, auc, mcc
    
    def _convert_if_not_tensor(self, x, dtype):
        if self._tensor_check(x):
            return x.to(self.device, dtype=dtype)
        else:
            return torch.Tensor(x, dtype=dtype, device=self.device)
    
    def _tensor_check(self, x):
        return isinstance(x, torch.Tensor)