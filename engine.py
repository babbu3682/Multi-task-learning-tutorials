import os
import math
import utils
import numpy as np
import torch
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt

from metrics import *
from losses import soft_dice_score
from sklearn.metrics import roc_auc_score


def freeze_params(model: torch.nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def unfreeze_params(model: torch.nn.Module):
    """Set requires_grad=True for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = True

def predict(self, x):
    """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

    Args:
        x: 4D torch tensor with shape (batch_size, channels, height, width)

    Return:
        prediction: 4D torch tensor with shape (batch_size, classes, height, width)

    """
    if self.training:
        self.eval()

    with torch.no_grad():
        x = self.forward(x)

    return x


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

features   = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook


def Activation_Map(x):
    # print("mean 0 == ", x.shape)                                # x = (B, 2048, 16, 16, D)
    mean = torch.mean(x, dim=1, keepdim=True)                     # x = (B, 1, H, W ,D)
    mean = torch.sigmoid(mean).squeeze().detach().cpu().numpy()   # x = (H, W, D)
    mean = np.stack([ cv2.resize(mean[..., i], (256, 256), interpolation=cv2.INTER_CUBIC) for i in range(mean.shape[-1]) ], axis=-1)
    mean -= mean.min()
    mean /= mean.max()
    return torch.tensor(mean).unsqueeze(0)


########################################################
# Uptask Task
    # SMART_Net
def train_Up_SMART_Net(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=inputs)
        
        if hasattr(model, 'module'):
            consistency_loss = F.mse_loss(cls_pred, model.module.gem_pool(seg_pred).flatten(1))
        else: 
            consistency_loss = F.mse_loss(cls_pred, model.gem_pool(seg_pred).flatten(1))
        

        loss += 0.5*consistency_loss
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("image == ", batch_data['image_meta_dict']['filename_or_obj'])
            print("label == ", batch_data['label_meta_dict']['filename_or_obj'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
            metric_logger.update(Consist_Loss=0.5*consistency_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Up_SMART_Net(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    total_cls_pred  = torch.tensor([])
    total_cls_true  = torch.tensor([])
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, seg_pred, rec_pred = model(inputs)


        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=inputs)
        if hasattr(model, 'module'):
            consistency_loss = F.mse_loss(cls_pred, model.module.gem_pool(seg_pred).flatten(1))
        else: 
            consistency_loss = F.mse_loss(cls_pred, model.gem_pool(seg_pred).flatten(1))

        loss += 0.5*consistency_loss        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)        
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
            metric_logger.update(Consist_Loss=0.5*consistency_loss.item())

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)
        seg_pred = torch.sigmoid(seg_pred)

        total_cls_pred  = torch.cat([total_cls_pred, cls_pred.detach().cpu()])
        total_cls_true  = torch.cat([total_cls_true, cls_gt.detach().cpu()])

        # Metrics SEG
        if seg_gt.any():
            dice = soft_dice_score(output=seg_pred.round(), target=seg_gt, smooth=0.0)    # pred_seg must be round() !! 
            metric_logger.update(dice=dice.item())     

        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

    # Metric CLS
    auc            = roc_auc_score(y_true=total_cls_true, y_score=total_cls_pred)
    tp, fp, fn, tn = get_stats(total_cls_pred.round().long(), total_cls_true.long(), mode="binary")        
    f1             = f1_score(tp, fp, fn, tn, reduction="macro")
    acc            = accuracy(tp, fp, fn, tn, reduction="macro")
    sen            = sensitivity(tp, fp, fn, tn, reduction="macro")
    spe            = specificity(tp, fp, fn, tn, reduction="macro")

    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
    
@torch.no_grad()
def test_Up_SMART_Net(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    total_cls_pred  = torch.tensor([])
    total_cls_true  = torch.tensor([])

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=inputs)
        if hasattr(model, 'module'):
            consistency_loss = F.mse_loss(cls_pred, model.module.gem_pool(seg_pred).flatten(1))
        else: 
            consistency_loss = F.mse_loss(cls_pred, model.gem_pool(seg_pred).flatten(1))

        loss += 0.5*consistency_loss        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
            metric_logger.update(Consist_Loss=0.5*consistency_loss.item())

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)
        seg_pred = torch.sigmoid(seg_pred)

        total_cls_pred  = torch.cat([total_cls_pred, cls_pred.detach().cpu()])
        total_cls_true  = torch.cat([total_cls_true, cls_gt.detach().cpu()])

        # Metrics SEG
        if seg_gt.any():
            dice = soft_dice_score(output=seg_pred.round(), target=seg_gt, smooth=0.0)    # pred_seg must be round() !! 
            metric_logger.update(dice=dice.item())     
  
        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

    # Metric CLS
    auc            = roc_auc_score(y_true=total_cls_true, y_score=total_cls_pred)
    tp, fp, fn, tn = get_stats(total_cls_pred.round().long(), total_cls_true.long(), mode="binary")        
    f1             = f1_score(tp, fp, fn, tn, reduction="macro")
    acc            = accuracy(tp, fp, fn, tn, reduction="macro")
    sen            = sensitivity(tp, fp, fn, tn, reduction="macro")
    spe            = specificity(tp, fp, fn, tn, reduction="macro")

    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)     

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
     





    # SMART_Net
def train_Up_SMART_Net_V2(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, seg_pred, rec_pred, _, _ = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=inputs)
        
        if hasattr(model, 'module'):
            consistency_loss = F.mse_loss(cls_pred, model.module.gem_pool(seg_pred).flatten(1))
        else: 
            consistency_loss = F.mse_loss(cls_pred, model.gem_pool(seg_pred).flatten(1))
        

        loss += 0.5*consistency_loss
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("image == ", batch_data['image_meta_dict']['filename_or_obj'])
            print("label == ", batch_data['label_meta_dict']['filename_or_obj'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
            metric_logger.update(Consist_Loss=0.5*consistency_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Up_SMART_Net_V2(model, criterion, data_loader, device, print_freq, batch_size, epoch, png_save_dir):
    os.makedirs(png_save_dir, mode=0o777, exist_ok=True)
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    total_cls_pred  = torch.tensor([])
    total_cls_true  = torch.tensor([])
    pos_save_png = True
    neg_save_png = True

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, seg_pred, rec_pred, x_distort, noise_mask = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=inputs)
        if hasattr(model, 'module'):
            consistency_loss = F.mse_loss(cls_pred, model.module.gem_pool(seg_pred).flatten(1))
        else: 
            consistency_loss = F.mse_loss(cls_pred, model.gem_pool(seg_pred).flatten(1))

        loss += 0.5*consistency_loss        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)        
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
            metric_logger.update(Consist_Loss=0.5*consistency_loss.item())

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)
        seg_pred = torch.sigmoid(seg_pred)

        total_cls_pred  = torch.cat([total_cls_pred, cls_pred.detach().cpu()])
        total_cls_true  = torch.cat([total_cls_true, cls_gt.detach().cpu()])

        # Metrics SEG
        if seg_gt.any():
            dice = soft_dice_score(output=seg_pred.round(), target=seg_gt, smooth=0.0)    # pred_seg must be round() !! 
            metric_logger.update(dice=dice.item())     

        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs)
        metric_logger.update(mae=mae.item())

        # Save
        if seg_gt.any() and pos_save_png:
            plt.imsave(png_save_dir+'/epoch_'+str(epoch)+'_p_x_distort.png', x_distort[0].detach().cpu().numpy().squeeze(), cmap="gray")    
            plt.imsave(png_save_dir+'/epoch_'+str(epoch)+'_p_rec_pred.png', rec_pred[0].detach().cpu().numpy().squeeze(), cmap="gray")
            plt.imsave(png_save_dir+'/epoch_'+str(epoch)+'_p_inputs.png', inputs[0].detach().cpu().numpy().squeeze(), cmap="gray")
            plt.imsave(png_save_dir+'/epoch_'+str(epoch)+'_p_seg_mask.png', seg_gt[0].detach().cpu().numpy().squeeze(), cmap="gray")
            plt.imsave(png_save_dir+'/epoch_'+str(epoch)+'_p_cls_pred{'+str(round(cls_pred.item(), 3))+'}_p_seg_pred.png', seg_pred[0].round().detach().cpu().numpy().squeeze(), cmap="gray")
            
            plt.imshow(noise_mask[0].detach().cpu().numpy().squeeze(), cmap='jet')
            plt.colorbar()
            plt.savefig(png_save_dir+'/epoch_'+str(epoch)+'_p_noise_mask.png')
            plt.close(); plt.cla(); plt.clf()
            pos_save_png = False

        if not seg_gt.any() and neg_save_png:
            plt.imsave(png_save_dir+'/epoch_'+str(epoch)+'_n_x_distort.png', x_distort[0].detach().cpu().numpy().squeeze(), cmap="gray")    
            plt.imsave(png_save_dir+'/epoch_'+str(epoch)+'_n_rec_pred.png', rec_pred[0].detach().cpu().numpy().squeeze(), cmap="gray")
            plt.imsave(png_save_dir+'/epoch_'+str(epoch)+'_n_inputs.png', inputs[0].detach().cpu().numpy().squeeze(), cmap="gray")
            plt.imsave(png_save_dir+'/epoch_'+str(epoch)+'_n_seg_mask.png', seg_gt[0].detach().cpu().numpy().squeeze(), cmap="gray")
            plt.imsave(png_save_dir+'/epoch_'+str(epoch)+'_n_cls_pred{'+str(round(cls_pred.item(), 3))+'}_n_seg_pred.png', seg_pred[0].round().detach().cpu().numpy().squeeze(), cmap="gray")
            
            plt.imshow(noise_mask[0].detach().cpu().numpy().squeeze(), cmap='jet')
            plt.colorbar()
            plt.savefig(png_save_dir+'/epoch_'+str(epoch)+'_n_noise_mask.png')
            plt.close(); plt.cla(); plt.clf()
            neg_save_png = False

    # Metric CLS
    auc            = roc_auc_score(y_true=total_cls_true, y_score=total_cls_pred)
    tp, fp, fn, tn = get_stats(total_cls_pred.round().long(), total_cls_true.long(), mode="binary")        
    f1             = f1_score(tp, fp, fn, tn, reduction="macro")
    acc            = accuracy(tp, fp, fn, tn, reduction="macro")
    sen            = sensitivity(tp, fp, fn, tn, reduction="macro")
    spe            = specificity(tp, fp, fn, tn, reduction="macro")

    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
    
@torch.no_grad()
def test_Up_SMART_Net_V2(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    total_cls_pred  = torch.tensor([])
    total_cls_true  = torch.tensor([])

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=inputs)
        if hasattr(model, 'module'):
            consistency_loss = F.mse_loss(cls_pred, model.module.gem_pool(seg_pred).flatten(1))
        else: 
            consistency_loss = F.mse_loss(cls_pred, model.gem_pool(seg_pred).flatten(1))

        loss += 0.5*consistency_loss        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
            metric_logger.update(Consist_Loss=0.5*consistency_loss.item())

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)
        seg_pred = torch.sigmoid(seg_pred)

        total_cls_pred  = torch.cat([total_cls_pred, cls_pred.detach().cpu()])
        total_cls_true  = torch.cat([total_cls_true, cls_gt.detach().cpu()])

        # Metrics SEG
        if seg_gt.any():
            dice = soft_dice_score(output=seg_pred.round(), target=seg_gt, smooth=0.0)    # pred_seg must be round() !! 
            metric_logger.update(dice=dice.item())     
  
        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

    # Metric CLS
    auc            = roc_auc_score(y_true=total_cls_true, y_score=total_cls_pred)
    tp, fp, fn, tn = get_stats(total_cls_pred.round().long(), total_cls_true.long(), mode="binary")        
    f1             = f1_score(tp, fp, fn, tn, reduction="macro")
    acc            = accuracy(tp, fp, fn, tn, reduction="macro")
    sen            = sensitivity(tp, fp, fn, tn, reduction="macro")
    spe            = specificity(tp, fp, fn, tn, reduction="macro")

    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)     

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
     












########################################################
# Down Task
    # CLS
def train_Down_SMART_Net_CLS(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size, gradual_unfreeze):
    # 3d patient-level based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # print("Unfreeze encoder.stem ...!")
    # unfreeze_params(model.module.encoder._blocks) if hasattr(model, 'module') else unfreeze_params(model.encoder._blocks)

    # EfficientNet
    if gradual_unfreeze:
        if epoch >= 0 and epoch <= 100:
            print("Freeze encoder ...!")
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
        elif epoch >= 101 and epoch < 111:
            print("Unfreeze encoder.layer4 ...!")
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
            unfreeze_params(model.module.encoder._blocks[model.module.encoder._stage_idxs[2] :]) if hasattr(model, 'module') else unfreeze_params(model.encoder._blocks[model.encoder._stage_idxs[2] :])        
        elif epoch >= 111 and epoch < 121:
            print("Unfreeze encoder.layer3 ...!")
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
            unfreeze_params(model.module.encoder._blocks[model.module.encoder._stage_idxs[1] : model.module.encoder._stage_idxs[2]]) if hasattr(model, 'module') else unfreeze_params(model.encoder._blocks[model.encoder._stage_idxs[1] : model.encoder._stage_idxs[2]])
        elif epoch >= 121 and epoch < 131:
            print("Unfreeze encoder.layer2 ...!")
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
            unfreeze_params(model.module.encoder._blocks[model.module.encoder._stage_idxs[2] : model.module.encoder._stage_idxs[3]]) if hasattr(model, 'module') else unfreeze_params(model.encoder._blocks[model.encoder._stage_idxs[1] : model.encoder._stage_idxs[2]])
        elif epoch >= 131 and epoch < 141:
            print("Unfreeze encoder.layer1 ...!")
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
            unfreeze_params(model.module.encoder._blocks[model.module.encoder._stage_idxs[3] : model.module.encoder._stage_idxs[4]]) if hasattr(model, 'module') else unfreeze_params(model.encoder._blocks[model.encoder._stage_idxs[1] : model.encoder._stage_idxs[2]])
        else :
            # print("Unfreeze encoder.stem ...!")
            # unfreeze_params(model.module.encoder) if hasattr(model, 'module') else unfreeze_params(model.encoder)
            print("Unfreeze encoder.layer1 ...!")
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
            unfreeze_params(model.module.encoder._blocks[model.module.encoder._stage_idxs[3] : model.module.encoder._stage_idxs[4]]) if hasattr(model, 'module') else unfreeze_params(model.encoder._blocks[model.encoder._stage_idxs[1] : model.encoder._stage_idxs[2]])            
    else :
        print("Freeze encoder ...!")
        freeze_params(model.module.encoder._blocks) if hasattr(model, 'module') else freeze_params(model.encoder._blocks)

    # # ResNet
    # if gradual_unfreeze:
    #     # Gradual Unfreezing
    #     # 10 epoch 씩 one stage block 풀기, 100 epoch까지는 아예 고정
    #     if epoch >= 0 and epoch <= 100:
    #         freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
    #         print("Freeze encoder ...!")
    #     elif epoch >= 101 and epoch < 111:
    #         print("Unfreeze encoder.layer4 ...!")
    #         unfreeze_params(model.module.encoder.layer4) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer4)
    #     elif epoch >= 111 and epoch < 121:
    #         print("Unfreeze encoder.layer3 ...!")
    #         unfreeze_params(model.module.encoder.layer3) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer3)
    #     elif epoch >= 121 and epoch < 131:
    #         print("Unfreeze encoder.layer2 ...!")
    #         unfreeze_params(model.module.encoder.layer2) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer2)
    #     elif epoch >= 131 and epoch < 141:
    #         print("Unfreeze encoder.layer1 ...!")
    #         unfreeze_params(model.module.encoder.layer1) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer1)
    #     else :
    #         print("Unfreeze encoder.stem ...!")
    #         unfreeze_params(model.module.encoder) if hasattr(model, 'module') else unfreeze_params(model.encoder)
    # else :
    #     print("Freeze encoder ...!")
    #     freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                                                        # (B, C, H, W, D)
        cls_gt  = batch_data["label"].flatten(1).bool().any(dim=1, keepdim=True).float().to(device)     #  ---> (B, 1)
        depths  = batch_data["depths"]                                                                  #  ---> (B, 1) Fix bug, change cpu()

        cls_pred = model(inputs, depths)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Down_SMART_Net_CLS(model, criterion, data_loader, device, print_freq, batch_size):
    # 3d patient-level based evaluate ...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    total_cls_pred  = torch.tensor([])
    total_cls_true  = torch.tensor([])

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                                                        # (B, C, H, W, D)
        cls_gt  = batch_data["label"].flatten(1).bool().any(dim=1, keepdim=True).float().to(device)     #  ---> (B, 1)
        depths  = batch_data["depths"]                                                                  #  ---> (B, 1) Fix bug, change cpu()

        cls_pred = model(inputs, depths)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # Post-processing
        cls_pred = torch.sigmoid(cls_pred)

        total_cls_pred  = torch.cat([total_cls_pred, cls_pred.detach().cpu()])
        total_cls_true  = torch.cat([total_cls_true, cls_gt.detach().cpu()])

    # Metric CLS
    auc            = roc_auc_score(y_true=total_cls_true, y_score=total_cls_pred)
    tp, fp, fn, tn = get_stats(total_cls_pred.round().long(), total_cls_true.long(), mode="binary")        
    f1             = f1_score(tp, fp, fn, tn, reduction="macro")
    acc            = accuracy(tp, fp, fn, tn, reduction="macro")
    sen            = sensitivity(tp, fp, fn, tn, reduction="macro")
    spe            = specificity(tp, fp, fn, tn, reduction="macro")

    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)    

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_Down_SMART_Net_CLS(model, criterion, data_loader, device, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'
    
    total_cls_pred  = torch.tensor([])
    total_cls_true  = torch.tensor([])

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                                                        # (B, C, H, W, D)
        cls_gt  = batch_data["label"].flatten(1).bool().any(dim=1, keepdim=True).float().to(device)     #  ---> (B, 1)
        depths  = batch_data["depths"]                                                                  #  ---> (B, 1) Fix bug, change cpu()
        
        cls_pred = model(inputs, depths)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # Post-processing
        cls_pred = torch.sigmoid(cls_pred)

        total_cls_pred  = torch.cat([total_cls_pred, cls_pred.detach().cpu()])
        total_cls_true  = torch.cat([total_cls_true, cls_gt.detach().cpu()])      


    # Metric CLS
    auc            = roc_auc_score(y_true=total_cls_true, y_score=total_cls_pred)
    tp, fp, fn, tn = get_stats(total_cls_pred.round().long(), total_cls_true.long(), mode="binary")        
    f1             = f1_score(tp, fp, fn, tn, reduction="macro")
    acc            = accuracy(tp, fp, fn, tn, reduction="macro")
    sen            = sensitivity(tp, fp, fn, tn, reduction="macro")
    spe            = specificity(tp, fp, fn, tn, reduction="macro")

    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)     

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

    # SEG
def train_Down_SMART_Net_SEG(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size, gradual_unfreeze):
    # 3d patient-level based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    

    # EfficientNet
    if gradual_unfreeze:
        if epoch >= 0 and epoch <= 100:
            print("Freeze encoder ...!")
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
        elif epoch >= 101 and epoch < 111:
            print("Unfreeze encoder.layer4 ...!")
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
            unfreeze_params(model.module.encoder._blocks[model.module.encoder._stage_idxs[2] :]) if hasattr(model, 'module') else unfreeze_params(model.encoder._blocks[model.encoder._stage_idxs[2] :])        
        elif epoch >= 111 and epoch < 121:
            print("Unfreeze encoder.layer3 ...!")
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
            unfreeze_params(model.module.encoder._blocks[model.module.encoder._stage_idxs[1] : model.module.encoder._stage_idxs[2]]) if hasattr(model, 'module') else unfreeze_params(model.encoder._blocks[model.encoder._stage_idxs[1] : model.encoder._stage_idxs[2]])
        elif epoch >= 121 and epoch < 131:
            print("Unfreeze encoder.layer2 ...!")
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
            unfreeze_params(model.module.encoder._blocks[model.module.encoder._stage_idxs[2] : model.module.encoder._stage_idxs[3]]) if hasattr(model, 'module') else unfreeze_params(model.encoder._blocks[model.encoder._stage_idxs[1] : model.encoder._stage_idxs[2]])
        elif epoch >= 131 and epoch < 141:
            print("Unfreeze encoder.layer1 ...!")
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
            unfreeze_params(model.module.encoder._blocks[model.module.encoder._stage_idxs[3] : model.module.encoder._stage_idxs[4]]) if hasattr(model, 'module') else unfreeze_params(model.encoder._blocks[model.encoder._stage_idxs[1] : model.encoder._stage_idxs[2]])
        else :
            # print("Unfreeze encoder.stem ...!")
            # unfreeze_params(model.module.encoder) if hasattr(model, 'module') else unfreeze_params(model.encoder)
            print("Unfreeze encoder.layer1 ...!")
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
            unfreeze_params(model.module.encoder._blocks[model.module.encoder._stage_idxs[3] : model.module.encoder._stage_idxs[4]]) if hasattr(model, 'module') else unfreeze_params(model.encoder._blocks[model.encoder._stage_idxs[1] : model.encoder._stage_idxs[2]])
    else :
        print("Freeze encoder ...!")
        freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)

    # # ResNet
    # if gradual_unfreeze:
    #     # Gradual Unfreezing
    #     # 10 epoch 씩 one stage block 풀기, 100 epoch까지는 아예 고정
    #     if epoch >= 0 and epoch <= 100:
    #         freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
    #         print("Freeze encoder ...!")
    #     elif epoch >= 101 and epoch < 111:
    #         print("Unfreeze encoder.layer4 ...!")
    #         unfreeze_params(model.module.encoder.layer4) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer4)
    #     elif epoch >= 111 and epoch < 121:
    #         print("Unfreeze encoder.layer3 ...!")
    #         unfreeze_params(model.module.encoder.layer3) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer3)
    #     elif epoch >= 121 and epoch < 131:
    #         print("Unfreeze encoder.layer2 ...!")
    #         unfreeze_params(model.module.encoder.layer2) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer2)
    #     elif epoch >= 131 and epoch < 141:
    #         print("Unfreeze encoder.layer1 ...!")
    #         unfreeze_params(model.module.encoder.layer1) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer1)
    #     else :
    #         print("Unfreeze encoder.stem ...!")
    #         unfreeze_params(model.module.encoder) if hasattr(model, 'module') else unfreeze_params(model.encoder)
    # else :
    #     print("Freeze encoder ...!")
    #     freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                 # (B, C, H, W, D)
        seg_gt  = batch_data["label"].to(device)                 # (B, C, H, W, D)

        seg_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Down_SMART_Net_SEG(model, criterion, data_loader, device, print_freq, batch_size):
    # 3d patient-level based evaluate ...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                                   # (B, C, H, W, D)
        seg_gt  = batch_data["label"].to(device)                                   # (B, C, H, W, D)

        seg_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        seg_pred = torch.sigmoid(seg_pred)

        # Metrics SEG
        if seg_gt.any():
            dice = soft_dice_score(output=seg_pred.round(), target=seg_gt, smooth=0.0)    # pred_seg must be round() !! 
            metric_logger.update(dice=dice.item())     

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_Down_SMART_Net_SEG(model, criterion, data_loader, device, print_freq, batch_size):
    # 3d patient-level based evaluate ...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                                   # (B, C, H, W, D)
        seg_gt  = batch_data["label"].to(device)                                   # (B, C, H, W, D)

        seg_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        seg_pred = torch.sigmoid(seg_pred)

        # Metrics SEG
        if seg_gt.any():
            dice = soft_dice_score(output=seg_pred.round(), target=seg_gt, smooth=0.0)    # pred_seg must be round() !! 
            metric_logger.update(dice=dice.item())      

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}





####### Patient CLS & SEG
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().cpu()
    return hook

@torch.no_grad()
def test_Patient_Cls_model(model, criterion, data_loader, device, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'
    
    total_cls_pred  = torch.tensor([])
    total_cls_true  = torch.tensor([])

    save_folder = '/workspace/sunggu/1.Hemorrhage/SMART-Net-Upgrade/checkpoints/version_2_model/coreline_cls/'
    act_image_saver = SaveImage(output_dir=save_folder, 
                                output_postfix='Act_Cls', 
                                output_ext='.nii.gz', 
                                resample=False, 
                                mode='bilinear', 
                                squeeze_end_dims=True, 
                                data_root_dir='', 
                                separate_folder=False, 
                                print_log=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                                                        # (B, C, H, W, D)
        cls_gt  = batch_data["label"].flatten(1).bool().any(dim=1, keepdim=True).float().to(device)     #  ---> (B, 1)
        depths  = batch_data["depths"]   
        image_save_dict = batch_data['image_meta_dict'][0]                                                               #  ---> (B, 1) Fix bug, change cpu()
        
        cls_pred = model(inputs, depths)
        act_pred = model.extract_feat(inputs, depths)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # Post-processing
        cls_pred = torch.sigmoid(cls_pred)

        total_cls_pred  = torch.cat([total_cls_pred, cls_pred.detach().cpu()])
        total_cls_true  = torch.cat([total_cls_true, cls_gt.detach().cpu()])    

        # CAM Save nii.gz
        #   Resize 512 x 512
        act_map   = Activation_Map(act_pred)
        act_map   = Resize(spatial_size=(512, 512, act_map.shape[-1]),   mode='trilinear', align_corners=True)(act_map)        # Input = (C, H, W, D)

        #   Orientation
        act_map   = Flip(spatial_axis=1)(act_map)
        act_map   = Rotate90(k=1, spatial_axes=(0, 1))(act_map) 

        act_image_saver((act_map.numpy()>=0.8), image_save_dict)           # Note: image should be channel-first shape: [C,H,W,[D]].

        os.rename(save_folder+image_save_dict['filename_or_obj'].split('/')[-1].split('.')[0]+'_Act_Cls.nii.gz', save_folder+image_save_dict['filename_or_obj'].split('/')[-1].split('.')[0]+'_Act_Cls_['+str(round(cls_pred.squeeze().item(), 4))+'].nii.gz')

    # Metric CLS
    auc            = roc_auc_score(y_true=total_cls_true, y_score=total_cls_pred)
    tp, fp, fn, tn = get_stats(total_cls_pred.round().long(), total_cls_true.long(), mode="binary")        
    f1             = f1_score(tp, fp, fn, tn, reduction="macro")
    acc            = accuracy(tp, fp, fn, tn, reduction="macro")
    sen            = sensitivity(tp, fp, fn, tn, reduction="macro")
    spe            = specificity(tp, fp, fn, tn, reduction="macro")

    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)     

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_Patient_Seg_model(model, criterion, data_loader, device, print_freq, batch_size):
    # 3d patient-level based evaluate ...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'
    dice_list = []

    input_image_saver = SaveImage(output_dir='/workspace/sunggu/1.Hemorrhage/SMART-Net-Upgrade/checkpoints/version_2_model/coreline_demo_seg/', 
                                output_postfix='Input_Seg', 
                                output_ext='.nii.gz', 
                                resample=False, 
                                mode='bilinear', 
                                squeeze_end_dims=True, 
                                data_root_dir='', 
                                separate_folder=False, 
                                print_log=True)

    pred_mask_saver = SaveImage(output_dir='/workspace/sunggu/1.Hemorrhage/SMART-Net-Upgrade/checkpoints/version_2_model/coreline_demo_seg/', 
                                output_postfix='Pred_Seg', 
                                output_ext='.nii.gz', 
                                resample=False, 
                                mode='nearest', 
                                squeeze_end_dims=True, 
                                data_root_dir='', 
                                separate_folder=False, 
                                print_log=True)


    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                                   # (B, C, H, W, D)
        seg_gt  = batch_data["label"].to(device)                                   # (B, C, H, W, D)
        image_save_dict = batch_data['image_meta_dict'][0]
        mask_save_dict  = batch_data['label_meta_dict'][0]

        seg_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        seg_pred = torch.sigmoid(seg_pred)
        
        # Metrics SEG
        if seg_gt.any():
            dice = soft_dice_score(output=seg_pred.round(), target=seg_gt, smooth=0.0)    # pred_seg must be round() !! 
            metric_logger.update(dice=dice.item())
            dice_list.append(dice.item())      

        # Save nii.gz
        #   Resize 512 x 512
        inputs   = Resize(spatial_size=(512, 512, inputs.shape[-1]),   mode='trilinear', align_corners=True)(inputs.detach().cpu().squeeze(0))           # Input = (C, H, W, D)
        seg_pred = Resize(spatial_size=(512, 512, seg_pred.shape[-1]), mode='nearest', align_corners=None)(seg_pred.detach().cpu().squeeze(0).round())   # Input = (C, H, W, D)

        #   Orientation
        inputs   = Flip(spatial_axis=1)(inputs)
        inputs   = Rotate90(k=1, spatial_axes=(0, 1))(inputs) 
        seg_pred = Flip(spatial_axis=1)(seg_pred)
        seg_pred = Rotate90(k=1, spatial_axes=(0, 1))(seg_pred) 

        input_image_saver(inputs.numpy(), image_save_dict)           # Note: image should be channel-first shape: [C,H,W,[D]].
        pred_mask_saver(seg_pred.numpy(), mask_save_dict)            # Note: image should be channel-first shape: [C,H,W,[D]].

    print("Dice Result == ", np.mean(dice_list))

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}





# Inference Patient version
@torch.no_grad()
def Infer_Patient_Cls_model(model, criterion, data_loader, device, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'
    
    save_folder = '/workspace/sunggu/1.Hemorrhage/SMART-Net-Upgrade/checkpoints/version_2_model/coreline_demo_cls/'
    act_image_saver = SaveImage(output_dir=save_folder, 
                                output_postfix='Act_Cls', 
                                output_ext='.nii.gz', 
                                resample=False, 
                                mode='bilinear', 
                                squeeze_end_dims=True, 
                                data_root_dir='', 
                                separate_folder=False, 
                                print_log=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                                                        # (B, C, H, W, D)
        depths  = batch_data["depths"]   
        image_save_dict = batch_data['image_meta_dict'][0]                                                               #  ---> (B, 1) Fix bug, change cpu()
        
        cls_pred = model(inputs, depths)
        act_pred = model.extract_feat(inputs, depths)

        # Post-processing
        cls_pred = torch.sigmoid(cls_pred)

        # CAM Save nii.gz
        #   Resize 512 x 512
        act_map   = Activation_Map(act_pred)
        act_map   = Resize(spatial_size=(512, 512, act_map.shape[-1]),   mode='trilinear', align_corners=True)(act_map)        # Input = (C, H, W, D)

        #   Orientation
        act_map   = Flip(spatial_axis=1)(act_map)
        act_map   = Rotate90(k=1, spatial_axes=(0, 1))(act_map) 
        
        # print("Name == ", image_save_dict['filename_or_obj'])
        # print("Pred == ", cls_pred)
        
        act_image_saver((act_map.numpy()>=0.8), image_save_dict)           # Note: image should be channel-first shape: [C,H,W,[D]].

        os.rename(save_folder+image_save_dict['filename_or_obj'].split('/')[-1].split('.')[0]+'_Act_Cls.nii.gz', save_folder+image_save_dict['filename_or_obj'].split('/')[-1].split('.')[0]+'_Act_Cls_['+str(round(cls_pred.squeeze().item(), 4))+'].nii.gz')

@torch.no_grad()
def Infer_Patient_Seg_model(model, criterion, data_loader, device, print_freq, batch_size):
    # 3d patient-level based evaluate ...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'
    dice_list = []

    input_image_saver = SaveImage(output_dir='/workspace/sunggu/1.Hemorrhage/SMART-Net-Upgrade/checkpoints/version_2_model/coreline_demo_seg/', 
                                output_postfix='Input_Seg', 
                                output_ext='.nii.gz', 
                                resample=False, 
                                mode='bilinear', 
                                squeeze_end_dims=True, 
                                data_root_dir='', 
                                separate_folder=False, 
                                print_log=True)

    pred_mask_saver = SaveImage(output_dir='/workspace/sunggu/1.Hemorrhage/SMART-Net-Upgrade/checkpoints/version_2_model/coreline_demo_seg/', 
                                output_postfix='Pred_Seg', 
                                output_ext='.nii.gz', 
                                resample=False, 
                                mode='nearest', 
                                squeeze_end_dims=True, 
                                data_root_dir='', 
                                separate_folder=False, 
                                print_log=True)


    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                                   # (B, C, H, W, D)
        image_save_dict = batch_data['image_meta_dict'][0]

        seg_pred = model(inputs)

        # post-processing
        seg_pred = torch.sigmoid(seg_pred)
        
        # Save nii.gz
        #   Resize 512 x 512
        inputs   = Resize(spatial_size=(512, 512, inputs.shape[-1]),   mode='trilinear', align_corners=True)(inputs.detach().cpu().squeeze(0))           # Input = (C, H, W, D)
        seg_pred = Resize(spatial_size=(512, 512, seg_pred.shape[-1]), mode='nearest', align_corners=None)(seg_pred.detach().cpu().squeeze(0).round())   # Input = (C, H, W, D)

        #   Orientation
        inputs   = Flip(spatial_axis=1)(inputs)
        inputs   = Rotate90(k=1, spatial_axes=(0, 1))(inputs) 
        seg_pred = Flip(spatial_axis=1)(seg_pred)
        seg_pred = Rotate90(k=1, spatial_axes=(0, 1))(seg_pred) 

        input_image_saver(inputs.numpy(), image_save_dict)           # Note: image should be channel-first shape: [C,H,W,[D]].
        pred_mask_saver(seg_pred.numpy(), image_save_dict)           # Note: image should be channel-first shape: [C,H,W,[D]].








########################################################
## Inference code 
from monai.transforms import SaveImage
from monai.transforms import Resize, Flip, Rotate90
from monai.utils import ensure_tuple
from typing import Dict, Optional, Union
import logging
import traceback


    # SMAET-Net
@torch.no_grad()
def infer_Up_SMART_Net(model, data_loader, device, print_freq, save_dir):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'TEST:'
    
    save_dict = dict()
    img_path_list = []
    img_list = []
    cls_list = []
    seg_list = []
    rec_list = []
    act_list = []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)

        model.encoder.layer4.register_forward_hook(get_activation('Activation Map')) # for Activation Map

        cls_pred, seg_pred, rec_pred = model(inputs)

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)
        seg_pred = torch.sigmoid(seg_pred)

        img_path_list.append(batch_data["image_path"][0])
        img_list.append(inputs.detach().cpu().squeeze())
        cls_list.append(cls_pred.detach().cpu().squeeze())
        seg_list.append(seg_pred.detach().cpu().squeeze())
        rec_list.append(rec_pred.detach().cpu().squeeze())
        act_list.append(Activation_Map(activation['Activation Map']))


    save_dict['img_path_list']  = img_path_list
    save_dict['img_list']       = img_list
    save_dict['cls_pred']       = cls_list
    save_dict['seg_pred']       = seg_list
    save_dict['rec_pred']       = rec_list
    save_dict['activation_map'] = act_list
    np.savez(save_dir + '/result.npz', result=save_dict) 

    # CLS
@torch.no_grad()
def infer_Down_SMART_Net_CLS(model, data_loader, device, print_freq, save_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'TEST:'

    save_dict = dict()
    img_path_list = []
    img_list  = []
    cls_list  = []
    feat_list = []


    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        
        inputs  = batch_data["image"].to(device)                                                        # (B, C, H, W, D)
        depths  = batch_data["depths"]                                                                  #  ---> (B, 1) Fix bug, change cpu()
        paths   = batch_data["image_path"][0]


        model.fc.register_forward_hook(get_features('feat')) # for Representation
        
        cls_pred = model(inputs, depths)

        # Post-processing
        cls_pred = torch.sigmoid(cls_pred)

        img_path_list.append(paths)
        img_list.append(inputs.detach().cpu().numpy().squeeze())
        cls_list.append(cls_pred.detach().cpu().numpy().squeeze())
        feat_list.append(features['feat'].detach().cpu().numpy().squeeze())


    save_dict['img_path_list']  = img_path_list
    save_dict['img_list']       = img_list
    save_dict['cls_pred']       = cls_list
    save_dict['feat']           = feat_list
    np.savez(save_dir + '/result.npz', result=save_dict) 


    # SEG
@torch.no_grad()
def infer_Down_SMART_Net_SEG(model, data_loader, device, print_freq, save_dir):
    # 3d patient-level based evaluate ...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'TEST:'

    pred_mask_saver = SaveImage(output_dir=save_dir, 
                                output_postfix='Pred', 
                                output_ext='.nii.gz', 
                                resample=True, 
                                mode='nearest', 
                                squeeze_end_dims=True, 
                                data_root_dir='', 
                                separate_folder=False, 
                                print_log=True)

    # Save npz path 
    save_dict = dict()

    img_path_list   = []
    img_list        = []
    seg_prob_list   = []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs    = batch_data["image"].to(device)   # (B, C, H, W, D)
        paths     = batch_data["image_path"][0]

        seg_pred = model(inputs)

        # post-processing
        seg_pred = torch.sigmoid(seg_pred)

        # Save
        img_path_list.append(paths)
        img_list.append(inputs.detach().cpu().numpy())
        seg_prob_list.append(seg_pred.round().detach().cpu().numpy())

        # resize 512 x 512
        inputs   = Resize(spatial_size=(512, 512, inputs.shape[-1]),   mode='trilinear', align_corners=True)(inputs.detach().cpu().numpy().squeeze(0)) # Input = (C, H, W, D)
        seg_pred = Resize(spatial_size=(512, 512, seg_pred.shape[-1]), mode='nearest', align_corners=None)(seg_pred.detach().cpu().numpy().squeeze(0).round()) # Input = (C, H, W, D)

        # Save nii        
        image_save_dict           = batch_data['image_meta_dict']
        image_save_dict['affine'] = image_save_dict['affine'].squeeze()
        image_save_dict['original_affine'] = image_save_dict['original_affine'].squeeze()

        print("C == ", image_save_dict.keys())
        # print("D == ", image_save_dict)
        # print("E == ", image_save_dict['affine'])
        print("E1 == ", image_save_dict['affine'].shape)
        print("E2 == ", image_save_dict['dim'])
        print("E3 == ", image_save_dict['original_affine'].shape)
        # print("E2 == ", image_save_dict['affine_np'])

        
        pred_mask_saver(inputs, image_save_dict)    # Note: image should be channel-first shape: [C,H,W,[D]].



    # Save Prediction by using npz
    save_dict['gt_img_path']   = img_path_list
    save_dict['gt_img']        = img_list
    save_dict['pred_mask']     = seg_prob_list

    np.savez(save_dir + '/result.npz', result=save_dict) 