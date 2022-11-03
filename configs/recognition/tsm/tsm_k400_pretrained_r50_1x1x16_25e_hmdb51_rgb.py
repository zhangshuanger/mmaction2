_base_ = [
    '../../_base_/models/tsm_r50.py', '../../_base_/schedules/sgd_tsm_50e.py',
    '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(num_segments=16),
    cls_head=dict(num_classes=31, num_segments=16),
    test_cfg=dict(average_clips='prob'))
#为得到 \[0, 1\] 间的动作分值
#model['test_cfg'] = dict(average_clips='prob')

# dataset settings
split = 1
dataset_type = 'RawframeDataset'
data_root = 'data/hmdb51/rawframes'
data_root_val = 'data/hmdb51/rawframes'
ann_file_train = f'data/hmdb51/hmdb51_train_split_{split}_rawframes.txt'
ann_file_val = f'data/hmdb51/hmdb51_val_split_{split}_rawframes.txt'
ann_file_test = f'data/hmdb51/hmdb51_val_split_{split}_rawframes.txt'
img_norm_cfg = dict( # 图像正则化参数设置
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False) # std图像正则化方差
    # mean图像正则化平均值

train_pipeline = [   # 训练数据前处理流水线步骤组成的列表
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),# SampleFrames 类的配置
    dict(type='RawFrameDecode'),# RawFrameDecode 类的配置 给定帧序列，加载对应帧，解码对应帧
    dict(type='Resize', scale=(-1, 256)),# 调整图片尺寸 scale调整比例
    dict(
        type='MultiScaleCrop',# 多尺寸裁剪，随机从一系列给定尺寸中选择一个比例尺寸进行裁剪
        input_size=224,# 网络输入
        scales=(1, 0.875, 0.75, 0.66),# 长宽比例选择范围
        random_crop=False,# 是否进行随机裁剪
        max_wh_scale_gap=1,# 长宽最大比例间隔
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),# 调整图片尺寸 scale调整比例 keep_ratio是否表示长宽比
    dict(type='Normalize', **img_norm_cfg),# 图片正则化  **img_norm_cfg图片正则化参数
    dict(type='FormatShape', input_format='NCHW'), # 将图片格式转变为给定的输入格式 NCHW最终的图片组成格式
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),# Collect 类决定哪些键会被传递到行为识别器中，key代表输入的键，meta_keys输入的元键
    dict(type='ToTensor', keys=['imgs', 'label'])# ToTensor 类将其他类型转化为 Tensor 类型，keys里是将被从其他类型转化为 Tensor 类型的特征
]
val_pipeline = [ # 验证数据前处理流水线步骤组成的列表
    dict(
        type='SampleFrames',# 选定采样哪些视频帧
        clip_len=1,# 每个输出视频片段的帧
        frame_interval=1,# 所采相邻帧的时序间隔
        num_clips=16,# 所采帧片段的数量
        test_mode=True),# 是否设置为测试模式采帧
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),# 中心裁剪,crop_size裁剪部分的尺寸
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [ # 测试数据前处理流水线步骤组成的列表
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict( # 数据的配置
    videos_per_gpu=6,# 单个 GPU 的批大小
    workers_per_gpu=2,# 单个 GPU 的 dataloader 的进程
    test_dataloader=dict(videos_per_gpu=1),# 测试过程 dataloader 的额外设置  videos_per_gpu单个 GPU 的批大小
    train=dict(# 训练数据集的设置
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(# 验证数据集的设置
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(# 测试数据集的设置
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

# optimizer优化器设置
# 构建优化器的设置，支持：
    # (1) 所有 PyTorch 原生的优化器，这些优化器的参数和 PyTorch 对应的一致；
    # (2) 自定义的优化器，这些优化器在 `constructor` 的基础上构建。
optimizer = dict(
    lr=0.00075,  # this lr is used for 8 gpus# 学习率
)
# learning policy学习策略设置
lr_config = dict(policy='step', step=[10, 20])# 用于注册学习率调整钩子的设置,policy调整器策略, 支持 CosineAnnealing，Cyclic等方法,step 学习率衰减步长
total_epochs = 25# 训练模型的总周期数
evaluation = dict(# 训练期间做验证的设置
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
    # interval执行验证的间隔,metrics验证方法

# runtime settings运行设置
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x16_50e_kinetics400_rgb/tsm_r50_256p_1x1x16_50e_kinetics400_rgb_20201010-85645c2a.pth'  # noqa: E501
# 从给定路径加载模型作为预训练模型. 这个选项不会用于断点恢复训练
work_dir = './work_dirs/tsm_hmdb51/'# 记录当前实验日志和模型权重文件的文件夹

