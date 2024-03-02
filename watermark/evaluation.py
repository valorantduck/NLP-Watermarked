import torch
from torch.utils.data import Dataset, DataLoader
import os, sys
sys.path.append(os.getcwd())
import torchvision.transforms as T
from PIL import Image
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
import numpy as np
from tqdm import tqdm
from detector import detect_watermark
from pytorch_fid import fid_score
import lpips
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from sklearn import metrics
import argparse


parser = argparse.ArgumentParser(description='Input relevant parameters. ')
parser.add_argument("--data_root", help="Root folder of the dataset", type=str)
parser.add_argument("--log_path", default="./vqgan_checkpoint", help="path of VQ-log", type=str)
parser.add_argument("--gamma", default=0.5, help="Proportions of the red and green list", type=float)
parser.add_argument("--delta", default=2., help="Parameter used to increase the green list logit", type=float)
parser.add_argument("--z_threshold", default=4., help="Watermark detection threshold z", type=float)
parser.add_argument('--eval_bs', default=10, help='Batchsize of the evaluation process', type=int)
parser.add_argument('--num_workers', default=15, help='num_workers of the evaluation process', type=int)
args = parser.parse_args()


class Evaluation_Dataset(Dataset):
    """
    加载数据集用于验证过程
    """
    def __init__(self, imgs_path):
        self.input_path = os.path.join(imgs_path, "input")
        self.w_watermark_path = os.path.join(imgs_path, "w_watermark")
        self.n_watermark_path = os.path.join(imgs_path, "n_watermark")
        imgs_names = os.listdir(self.input_path)
        self.input_imgs_dir = [os.path.join(self.input_path, img_name) 
                    for img_name in imgs_names 
                    if img_name.endswith(".png")]
        self.w_watermark_imgs_dir = [os.path.join(self.w_watermark_path, img_name) 
                    for img_name in imgs_names 
                    if img_name.endswith(".png")]
        self.n_watermark_imgs_dir = [os.path.join(self.n_watermark_path, img_name) 
                    for img_name in imgs_names 
                    if img_name.endswith(".png")]
        
    
    def preprocess(self, img_path):
        img = Image.open(img_path)
        img = T.ToTensor()(img)
        img = 2.*img - 1.   # 归一化

        return img
    
    def __len__(self):
        return len(self.input_imgs_dir)
    
    def __getitem__(self, index):
        input_path = self.input_imgs_dir[index]
        w_watermark_path = self.w_watermark_imgs_dir[index]
        n_watermark_path = self.n_watermark_imgs_dir[index]
        input_img = self.preprocess(input_path)
        w_watermark_img = self.preprocess(w_watermark_path)
        n_watermark_img = self.preprocess(n_watermark_path)
        return input_img, w_watermark_img, n_watermark_img


class Evaluation():
    def __init__(self, imgs_path=None, log_path=None, device=None, 
                 gamma=0.5, z_threshold=4.0, batch_size=14, num_workers=30):

        config_path = os.path.join(log_path, "configs/model.yaml")
        ckpt_path = os.path.join(log_path, "checkpoints/last.ckpt")
        self.device = device
        config = OmegaConf.load(config_path)
        self.model = self.load_vqgan(config, ckpt_path).to(device)

        self.lpips_metric = lpips.LPIPS(net='alex').to(self.device)
        self.gamma = gamma
        self.codebook_size = self.model.codebook_size
        self.z_threshold = z_threshold
        
        # 加载数据
        self.dataset_eval = Evaluation_Dataset(imgs_path)
        self.dataloader = DataLoader(self.dataset_eval, batch_size=batch_size, shuffle=False, 
                                     num_workers=num_workers, pin_memory=True)

        print("finish initation! ")
    
    def load_vqgan(self, config, ckpt_path):
        """
        加载VQGAN，用于图像量化
        """
        model = VQModel(**config.model.params)
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
        return model.eval()
    
    def watermark_detect(self, data):
        """
        对一个batch数据进行水印检测
        """
        B = len(data)
        z, _, [_, _, indices] = self.model.encode(data, add_watermark=False)     # 对图像重新量化
        indices = indices.view(B, -1)
        
        all_predict = []
        for bt_indices in indices:      # batch中每一个样本
            predict, _ = detect_watermark(bt_indices, self.gamma, 
                                       self.codebook_size, self.z_threshold)
            all_predict.append(predict)
        return all_predict

    def detect_all_watermark(self):
        """
        对所有数据进行水印检测
        """
        
        print("begin to calculate watermark detect accuracy! ")
        loop = tqdm(self.dataloader, leave=False, total=len(self.dataloader))
        detect_acc = []
        auc_all = []
        low_all = []
        for input_imgs, w_watermark_imgs, n_watermark_imgs in loop:
            w_watermark_imgs = w_watermark_imgs.to(self.device)
            n_watermark_imgs = n_watermark_imgs.to(self.device)
            w_watermark_predict = self.watermark_detect(w_watermark_imgs)
            n_watermark_predict = self.watermark_detect(n_watermark_imgs)
            n_watermark_detect = n_watermark_predict.count(False)   # 计算检测为无水印的个数
            w_watermark_detect = w_watermark_predict.count(True)    # 计算检测为有水印的个数
            
            preds = n_watermark_predict +  w_watermark_predict
            preds = [1-int(x) for x in preds]
            t_labels = [1] * len(n_watermark_predict) + [0] * len(w_watermark_predict)
            fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            low = tpr[np.where(fpr<.01)[0][-1]]
            
            auc_all.append(auc)
            low_all.append(low)
            detect_acc.append(n_watermark_detect/len(n_watermark_predict))
            detect_acc.append(w_watermark_detect/len(w_watermark_predict))
        final_acc = sum(detect_acc)/len(detect_acc)
        final_auc = sum(auc_all)/len(auc_all)
        final_low = sum(low_all)/len(low_all)
        
        print("finish calculating watermark detect accuracy! ")
        return final_acc, final_auc, final_low

    def calculate_fid(self):
        print("begin to calculate fid! ")
        
        input_folder = self.dataset_eval.input_path
        output_folder = self.dataset_eval.n_watermark_path
        n_watermark_fid = fid_score.calculate_fid_given_paths([input_folder, output_folder],
                                                 batch_size=16,
                                                 num_workers=30,
                                                 device=self.device,
                                                 dims=2048)     # 默认2048
        
        input_folder = self.dataset_eval.input_path
        output_folder = self.dataset_eval.w_watermark_path
        w_watermark_fid = fid_score.calculate_fid_given_paths([input_folder, output_folder],
                                                 batch_size=16,
                                                 num_workers=30,
                                                 device=self.device,
                                                 dims=2048)     # 默认2048
        
        print("finish calculating fid! ")
        return n_watermark_fid, w_watermark_fid
    
    def calculate_lpips(self):
        print("begin to calculate lpips! ")
        loop = tqdm(self.dataloader, leave=False, total=len(self.dataloader))
        lpips_d_n = []
        lpips_d_w = []
        for input_imgs, w_watermark_imgs, n_watermark_imgs in loop:
            input_imgs = input_imgs.to(self.device)
            n_watermark_imgs = n_watermark_imgs.to(self.device)
            w_watermark_imgs = w_watermark_imgs.to(self.device)
            with torch.no_grad():
                d_n = self.lpips_metric.forward(input_imgs, n_watermark_imgs)
                d_w = self.lpips_metric.forward(input_imgs, w_watermark_imgs)
            lpips_d_n.append(torch.sum(d_n).item()/len(d_n))
            lpips_d_w.append(torch.sum(d_w).item()/len(d_w))
        lpips_distances_n = sum(lpips_d_n) / len(lpips_d_n)
        lpips_distances_w = sum(lpips_d_w) / len(lpips_d_w)
        print("finish calculating lpips! ")
        return lpips_distances_n, lpips_distances_w
    
    def calculate_mse(self):
        print("begin to calculate mse! ")
        loop = tqdm(self.dataloader, leave=False, total=len(self.dataloader))
        mse_all_n = []
        mse_all_w = []
        for input_imgs, w_watermark_imgs, n_watermark_imgs in loop:
            input_imgs = input_imgs.to(self.device)
            n_watermark_imgs = n_watermark_imgs.to(self.device)
            w_watermark_imgs = w_watermark_imgs.to(self.device)
            mse_n = F.mse_loss(input_imgs, n_watermark_imgs)
            mse_w = F.mse_loss(input_imgs, w_watermark_imgs)
            mse_all_n.append(mse_n.mean().item())
            mse_all_w.append(mse_w.mean().item())
        mse_all_n = sum(mse_all_n) / len(mse_all_n)
        mse_all_w = sum(mse_all_w) / len(mse_all_w)
        print("finish calculating mse! ")
        return mse_all_n, mse_all_w
    
    def calculate_ms_ssim(self):
        """
        计算ssim
        """
        print("begin to calculate ms-ssim! ")
        loop = tqdm(self.dataloader, leave=False, total=len(self.dataloader))
        ms_ssim_all_n = []
        ms_ssim_all_w = []
        for input_imgs, w_watermark_imgs, n_watermark_imgs in loop:
            input_imgs = (input_imgs + 1) / 2
            n_watermark_imgs = (n_watermark_imgs + 1) / 2
            w_watermark_imgs = (w_watermark_imgs + 1) / 2
            input_imgs = input_imgs.to(self.device)
            n_watermark_imgs = n_watermark_imgs.to(self.device)
            w_watermark_imgs = w_watermark_imgs.to(self.device)
            ms_ssim_avg_n = ms_ssim(input_imgs, n_watermark_imgs, data_range=1, size_average=True)
            ms_ssim_avg_w = ms_ssim(input_imgs, w_watermark_imgs, data_range=1, size_average=True)
            ms_ssim_all_n.append(ms_ssim_avg_n)
            ms_ssim_all_w.append(ms_ssim_avg_w)
        ms_ssim_all_n = sum(ms_ssim_all_n) / len(ms_ssim_all_n)
        ms_ssim_all_w = sum(ms_ssim_all_w) / len(ms_ssim_all_w)
        print("finish calculating ms-ssim! ")
        return ms_ssim_all_n.item(), ms_ssim_all_w.item()
    
    def run(self):
        """
        执行评估过程
        """
        
        watermark_detect_acc, watermark_detect_auc, watermark_detect_low= self.detect_all_watermark()
        torch.cuda.empty_cache()
        
        n_watermark_fid, w_watermark_fid = self.calculate_fid()
        torch.cuda.empty_cache()
        
        lpips_distances_n, lpips_distances_w = self.calculate_lpips()
        torch.cuda.empty_cache()
        
        mse_all_n, mse_all_w = self.calculate_mse()
        torch.cuda.empty_cache()
        
        ms_ssim_n, ms_ssim_w = self.calculate_ms_ssim()
        torch.cuda.empty_cache()

        print("Watermark detection accuracy", watermark_detect_acc)
        print("Watermark detection AUC", watermark_detect_auc)
        print("TPR@1%FPR", watermark_detect_low)
        print("Unwatermarked image FID", n_watermark_fid)
        print("Watermarked image FID", w_watermark_fid)
        print("Unwatermarked image lpips distance", lpips_distances_n)
        print("Watermarked image lpips distance", lpips_distances_w)
        print("Unwatermarked image MSE", mse_all_n)
        print("Watermarked image MSE", mse_all_w)
        print("Unwatermarked image SSIM", ms_ssim_n)
        print("Watermarked image SSIM", ms_ssim_w)

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    imgs_path = args.data_root
    log_path = args.log_path
    gamma = args.gamma
    delta = args.delta
    z_threshold = args.z_threshold
    eval_bs = args.eval_bs
    eval_numworkers = args.num_workers

    evaluation = Evaluation(imgs_path=imgs_path, 
                            log_path=log_path, device=DEVICE, 
                            gamma=gamma, z_threshold=z_threshold, 
                            batch_size=eval_bs, num_workers=eval_numworkers)
    evaluation.run()
