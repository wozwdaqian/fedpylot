# FedPylot by Cyprien Quéméneur, GPL-3.0 license

import argparse
import os
from PIL import Image
import random
import shutil
from tqdm import tqdm
from datasets_utils import create_directories, archive_directories, get_distribution_dataframe, convert_bbox

PCB_TRAIN_SIZE = 10668
DEFAULT_CLASS_MAP = {       
    'missing_hole': 0,
    'mouse_bite': 1,
    'open_circuit': 2,
    'short': 3,
    'spur': 4,
    'spurious_copper': 5
}

CLASSID_to_NAME = {
    '0': 'missing_hole',
    '1': 'mouse_bite',
    '2': 'open_circuit',
    '3': 'short',
    '4': 'spur',
    '5': 'spurious_copper'
}



def get_iid_splits(nclients: int, val_frac: float) -> dict:
    """返回一个字典，用于存储PCB数据集中每个客户端的IID（独立同分布）划分。"""
    random.seed(0)  # 设置随机种子，以保证划分结果可重复
    client_frac = (1 - val_frac) / nclients  # 每个客户端所分配的数据比例
    indices = list(range(PCB_TRAIN_SIZE))  # 全部数据样本的索引列表
    client_split_size = int(PCB_TRAIN_SIZE * client_frac)  # 每个客户端所分配的数据样本数
    splits = {}

    # 创建客户端划分
    for k in range(1, nclients + 1):
        client_data = random.sample(indices, client_split_size)  # 随机抽取样本分配给客户端
        for index in client_data:
            splits[index] = f'client{k}'  # 将样本索引分配到客户端
            indices.remove(index)  # 从可用索引列表中移除已分配的样本

    # 剩余的数据分配给服务器，用于验证
    for index in indices:
        splits[index] = 'server'

    return splits


def process_pcb(img_path: str, label_path: str, target_path: str, data: str, class_map: dict, nclients: int,
                  val_frac: float, tar: bool) -> None:
    """转换 PCB 数据集的标注格式并将数据划分到服务器和客户端。"""
    print('正在转换标注并划分数据...')
    create_directories(target_path, nclients)  # 创建目标目录，用于存储服务器和各个客户端的数据
    splits = get_iid_splits(nclients, val_frac)  # 获取数据划分的映射信息，分配到客户端或服务器

    objects_distribution = get_distribution_dataframe(data, nclients)  # 初始化对象分布数据框，用于统计每个客户端和服务器的数据分布
    # 遍历 PCB 训练标签文件
    for fname in tqdm(os.listdir(label_path)):
        # 确定该文件所属的目标客户端或服务器目录
        destination = splits[int(fname[:-4])]
        objects_distribution.loc['Samples', destination] += 1  # 更新样本数量分布

        # 打开目标标签文件，准备写入转换后的标注信息
        with open(f'{target_path}/{destination}/labels/{fname}', 'w') as target_file:
            # 打开 PCB 标签文件
            with open(f'{label_path}/{fname}', 'r') as label_file:
                # 打开对应的 PCB 图像文件并获取图像的宽度和高度
                with open(f'{img_path}/{fname[:-3]}jpg', 'rb') as img_file:
                    img = Image.open(img_file)
                    img_width, img_height = img.size  # 获取图像尺寸
                
                # 将图像文件复制到目标目录中，不删除原文件
                shutil.copyfile(f'{img_path}/{fname[:-3]}jpg', f'{target_path}/{destination}/images/{fname[:-3]}jpg')

                # 遍历标签文件中的每一行（每个对象）
                for line in label_file.readlines():
                    line = line.split()
                    class_id, x, y, w, h = line  # 读取 YOLO 格式的类别ID和边界框坐标
                    obj_type = CLASSID_to_NAME[class_id]  # 从类别ID获取对应的类别名称

                    # 将转换后的标签信息写入目标文件
                    target_file.write(f'{class_id} {x} {y} {w} {h}\n')
                    # 更新对象分布信息
                    objects_distribution.loc[obj_type, destination] += 1
    
    # 保存对象分布统计信息到 CSV 文件
    objects_distribution.to_csv(f'{target_path}/objects_distribution.csv')
    
    # 如果 tar 参数为真，则将各客户端和服务器目录打包归档
    if tar:
        print('正在归档...')
        archive_directories(target_path, nclients)



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--img-path', type=str, default='pcb/Images', help='图像文件夹路径')
    args.add_argument('--label-path', type=str, default='pcb/labels/', help='标签文件夹路径')
    args.add_argument('--target-path', type=str, default='datasets/pcb', help='目标目录路径')
    args.add_argument('--data', type=str, default='data/pcb.yaml', help='数据配置文件的路径 (yaml 格式)')
    args.add_argument('--class-map', type=dict, default=DEFAULT_CLASS_MAP, help='注释和类别的映射，需与 yaml 文件匹配')
    args.add_argument('--nclients', type=int, default=2, help='联邦实验中的客户端数量')
    args.add_argument('--val-frac', type=float, default=0.25, help='服务器持有的验证数据比例')
    args.add_argument('--tar', action='store_true', help='是否归档联邦参与者的目录')
    args = args.parse_args()
    process_pcb(args.img_path, args.label_path, args.target_path, args.data, args.class_map, args.nclients, args.val_frac, args.tar)
