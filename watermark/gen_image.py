import torch
from torch.utils.data import Dataset, DataLoader
import os, sys
sys.path.append(os.getcwd())
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
import numpy as np
import shutil
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
import argparse

parser = argparse.ArgumentParser(description='Input relevant parameters. ')
parser.add_argument("--data_root", help="Root folder of the dataset", type=str)
parser.add_argument("--log_path", default="./vqgan_checkpoint", help="path of VQ-log", type=str)
parser.add_argument('--save_path', help='Save the images before and after quantization', type=str)
parser.add_argument('--imgsize', default=256, help='The image size used to perform the quantization process.', type=int)
parser.add_argument('--gen_bs', default=2, help='The batchsize of the watermarking and quantization process', type=int)
parser.add_argument('--test_num', default=None, help='The number of images used for evaluation', type=int)
parser.add_argument("--img_type", default="JPEG", help="Dataset image type", type=str)
parser.add_argument("--gamma", default=0.5, help="Proportions of the red and green list", type=float)
parser.add_argument("--delta", default=2., help="Parameter used to increase the green list logit", type=float)
args = parser.parse_args()


class VQGAN_Dataset(Dataset):
    """
    加载数据用于后续水印添加
    """
    def __init__(self, imgs_path, size, test_num=None, img_type=".jpg"):
        imgs_names = os.listdir(imgs_path)
        if test_num is not None:
            self.imgs_dir = [os.path.join(imgs_path, img_name) 
                        for img_name in imgs_names 
                        if img_name.endswith(img_type)][:test_num]
        else:
            self.imgs_dir = [os.path.join(imgs_path, img_name) 
                        for img_name in imgs_names 
                        if img_name.endswith(img_type)]
        self.img_size = size
    
    def preprocess(self, img_path):
        img = Image.open(img_path)
        
        # 直接缩放
        img = TF.resize(img, size=(self.img_size, self.img_size), interpolation=InterpolationMode.LANCZOS)
        
        img = T.ToTensor()(img)
        img = 2.*img - 1.   # 归一化
        if img.shape[0] == 1:
            img = img.repeat([3, 1, 1])
        return img
    
    def __len__(self):
        return len(self.imgs_dir)
    
    def __getitem__(self, index):
        img_path = self.imgs_dir[index]
        img = self.preprocess(img_path)
        return img


class Gen_Image():
    """
    用于将原图经过VQGAN后，生成无水印图像或加水印图像
    """
    def __init__(self, img_path, img_size, batch_size, log_path, gamma, delta,
                 device, save_path, test_num=None, img_type=".JPEG"):
        save_path = save_path
        self.input_path = os.path.join(save_path, "input")
        self.w_watermark_path = os.path.join(save_path, "w_watermark")
        self.n_watermark_path = os.path.join(save_path, "n_watermark")
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(self.input_path)
        os.makedirs(self.w_watermark_path)
        os.makedirs(self.n_watermark_path)
        
        self.gamma = gamma
        self.delta = delta

        config_path = os.path.join(log_path, "configs/model.yaml")
        ckpt_path = os.path.join(log_path, "checkpoints/last.ckpt")
        config = OmegaConf.load(config_path)
        self.model = self.load_vqgan(config, ckpt_path).to(device)
        self.img_count = 0
        
        dataset = VQGAN_Dataset(img_path, img_size, test_num=test_num, img_type=img_type)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=1, pin_memory=False)
        self.dataloader = dataloader
        self.device = device
        self.probability = []

        print("finish initation! ")
        
    def load_vqgan(self, config, ckpt_path):
        """
        加载VQGAN权重
        """
        model = VQModel(**config.model.params, gamma=self.gamma, delta=self.delta)
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
        print(1)
        return model.eval()
    
    def inference(self, data, add_watermark):
        """
        执行推理过程
        """
        z, _, [_, _, indices] = self.model.encode(data, add_watermark=add_watermark)
        xrec = self.model.decode(z)
        return xrec
    
    def to_pil(self, imgs):
        """
        图像转PIL格式
        """
        imgs = imgs.detach().cpu()
        imgs = torch.clamp(imgs, -1., 1.)
        imgs = (imgs + 1.)/2.
        imgs = imgs.permute(0, 2, 3, 1).numpy()
        imgs = (255*imgs).astype(np.uint8)
        return imgs
    
    def to_save_imgs(self, input_imgs, w_watermark_imgs, n_watermark_imgs):
        """
        保存图像
        """
        input_imgs = self.to_pil(input_imgs)
        w_watermark_imgs = self.to_pil(w_watermark_imgs)
        n_watermark_imgs = self.to_pil(n_watermark_imgs)

        for idx in range(len(input_imgs)):
            input_img = input_imgs[idx]
            input_img = Image.fromarray(input_img)
            if not input_img.mode == "RGB":
                input_img = input_img.convert("RGB")
                
            w_watermark_img = w_watermark_imgs[idx]
            w_watermark_img = Image.fromarray(w_watermark_img)
            if not w_watermark_img.mode == "RGB":
                w_watermark_img = w_watermark_img.convert("RGB")
                
            n_watermark_img = n_watermark_imgs[idx]
            n_watermark_img = Image.fromarray(n_watermark_img)
            if not n_watermark_img.mode == "RGB":
                n_watermark_img = n_watermark_img.convert("RGB")
            
            input_img.save(os.path.join(self.input_path, str("%06d"%self.img_count)+".png"))
            w_watermark_img.save(os.path.join(self.w_watermark_path, str("%06d"%self.img_count)+".png"))
            n_watermark_img.save(os.path.join(self.n_watermark_path, str("%06d"%self.img_count)+".png"))
            self.img_count += 1
        
    def run(self):
        """
        执行流程
        """
        loop = tqdm(self.dataloader, leave=False, total=len(self.dataloader))
        for input_imgs in loop:
            input_imgs = input_imgs.to(self.device)
            w_watermark_imgs = self.inference(input_imgs, add_watermark=True)
            n_watermark_imgs = self.inference(input_imgs, add_watermark=False)
            self.to_save_imgs(input_imgs, w_watermark_imgs, n_watermark_imgs)


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_path = args.data_root
    log_path = args.log_path
    save_path = args.save_path
    img_size = args.imgsize
    gen_bs = args.gen_bs
    test_num = args.test_num
    img_type = args.img_type
    gamma = args.gamma
    delta = args.delta
    
    
    gen_image = Gen_Image(img_path, img_size, gen_bs, log_path, gamma=gamma, delta=delta, device=DEVICE, 
                          save_path=save_path, img_type=img_type, test_num=test_num)
    gen_image.run()
