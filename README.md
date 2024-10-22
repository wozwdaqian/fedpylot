### 集群运行

#### 配置环境

这个你有经验，我就跳过了

#### 预处理数据


prepare_pcb.py的更多参数在文件中有说明作用。
其中nclients是客户端数量，val-frac是验证集比例，代码会把(1-val-frac)比例的数据集除于客户端数量，也就是(1-val-frac)/nclients的比例的数据集放到{target-path}/client{id}中，val-frac比例的数据集放到{target-path}/server中。

**你要提前设置好 PCB_TRAIN_SIZE、DEFAULT_CLASS_MAP、CLASSID_to_NAME**


```
python datasets/prepare_pcb.py --tar --nclients 1 --img-path pcb/Images --label-path pcb/labels/  --target-path datasets/pcb
```


#### 训练

下载模型
```
bash weights/get_weights.sh yolov7
```

client-rank代表使用哪个客户端的数据集进行训练，训练结果默认存放到run/train下

```
python yolov7/train.py     --client-rank 1     --epochs 50     --weights weights/yolov7/yolov7_training.pt     --data data/pcb.yaml     --batch 32     --img 640 640     --cfg yolov7/cfg/training/yolov7.yaml     --hyp data/hyps/hyp.scratch.clientopt.pcb.yaml     --workers 8
```


### 联邦学习



#### 配置环境

```
apt-get install build-essential fakeroot devscripts equivs
```



下载安装包，并安装
```
wget https://download.schedmd.com/slurm/slurm-24.05.3.tar.bz2
tar -xaf slurm-24.05.3.tar.bz2
cd slurm-24.05.3
mk-build-deps -i debian/control
debuild -b -uc -us
```


安装
```
sudo apt-get install -y slurm-wlm
```


配置密钥
```
sudo dd if=/dev/urandom bs=1 count=1024 > /etc/munge/munge.key
sudo chown root:root /etc/munge/munge.key
sudo chmod 400 /etc/munge/munge.key
sudo chown root:root  /etc/munge/
```



启动

只运行一次
```
sudo systemctl enable munge
sudo munged
```


重启就要重新运行
```
sudo slurmctld
sudo munged
sudo slurmd -N compute1
sudo slurmd -N compute2
sudo slurmd -N compute3
sinfo
```



#### 训练

```

```





