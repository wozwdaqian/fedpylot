# 下载预训练的权重文件
if [[ $SLURM_PROCID -eq 0 ]]; then       # 只在进程0上执行
    bash weights/get_weights.sh yolov7
fi

# 创建训练集和测试集，默认情况下，server文件夹下的为测试集，数据集存放路径看prepare_pcb.py，默认为datasets下的pcb文件夹
python datasets/prepare_pcb.py --tar

# 运行集中式实验（具体设置见 yolov7/train.py 文件）, --client-rank 1 表示使用client1中的数据作为训练集，训练结果保存路径为run/exp{id}
python yolov7/train.py \
    --client-rank 1 \
    --epochs 50 \
    --weights weights/yolov7/yolov7_training.pt \
    --data data/pcb.yaml \
    --batch 32 \
    --img 640 640 \
    --cfg yolov7/cfg/training/yolov7.yaml \
    --hyp data/hyps/hyp.scratch.clientopt.pcb.yaml \
    --workers 8 \

