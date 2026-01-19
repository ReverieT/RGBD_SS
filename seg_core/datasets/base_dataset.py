import os
import logging
from PIL import Image
from torch.utils.data import Dataset

class RGBXDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None, depth_ext='.png', label_ext='.png'):
        """
        Args:
            root_dir (str): 数据集根目录 (e.g. data/NYUDepthv2)
            split (str): 'train' 或 'test'/'val'
            transforms (callable): 组合好的数据增强
            depth_ext (str): 深度图后缀
            label_ext (str): 标签图后缀
        """
        self.root = root_dir
        self.split = split
        self.transforms = transforms
        self.depth_ext = depth_ext
        self.label_ext = label_ext
        
        # 读取文件列表
        # 假设 train.txt 位于 root_dir 下
        self.file_list = self._get_file_list(split)
        
        if len(self.file_list) == 0:
            logging.warning(f"Dataset at {root_dir} is empty for split {split}!")
        else:
            logging.info(f"Loaded {len(self.file_list)} samples for {split}.")

    def __getitem__(self, index):
        # 1. 获取基础文件名
        # train.txt line example: "RGB/0001.jpg"
        rgb_rel_path = self.file_list[index]
        
        # 提取不带后缀的文件名: "RGB/0001.jpg" -> "0001"
        basename = os.path.splitext(os.path.basename(rgb_rel_path))[0]
        
        # 2. 拼接绝对路径
        # 假设结构: 
        # root/RGB/name.jpg
        # root/Depth/name.png
        # root/Label/name.png
        rgb_path = os.path.join(self.root, rgb_rel_path)
        depth_path = os.path.join(self.root, 'Depth', basename + self.depth_ext)
        label_path = os.path.join(self.root, 'Label', basename + self.label_ext)

        # 3. 读取图像 (使用 PIL 以配合 transforms)
        try:
            # .convert('RGB') 确保是 3 通道
            image = Image.open(rgb_path).convert('RGB')
            # .convert('L') 确保是单通道灰度 (0-255)
            depth = Image.open(depth_path).convert('L')
            
            sample = {'image': image, 'depth': depth, 'name': basename}
            
            # 如果是训练集或验证集，需要读取 Label
            if self.split != 'test_inference': # 假设纯推理模式不需要Label
                if os.path.exists(label_path):
                    label = Image.open(label_path).convert('L')
                    sample['label'] = label
                else:
                    # 如果缺少 Label 文件但不是纯推理模式，可能会报错或忽略
                    # 这里为了健壮性，暂时忽略，但实际训练中应该报错
                    pass

            # 4. 数据增强 (Transforms)
            # 这里的 transforms 会处理 dict 内的所有图像，保证同步
            if self.transforms:
                sample = self.transforms(sample)
                
            return sample

        except Exception as e:
            # 打印报错的文件路径，方便调试
            print(f"Error loading sample: {rgb_path}")
            raise e

    def __len__(self):
        return len(self.file_list)

    def _get_file_list(self, split):
        """读取 txt 索引文件"""
        list_path = os.path.join(self.root, f'{split}.txt')
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"List file not found: {list_path}")
            
        file_list = []
        with open(list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 这里假设 txt 里已经是 "RGB/xxxx.jpg" 这种格式
                    # 如果 txt 里包含 Label 路径 (如 "RGB/1.jpg Label/1.png")，取第一部分
                    file_list.append(line.split()[0]) 
        return file_list