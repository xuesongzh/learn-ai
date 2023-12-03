from rknn.api import RKNN

if __name__ == '__main__':
    # 使用RKNN方法创建RKNN对象
    rknn = RKNN()

    # 使用config接口配置要生成的RKNN对象
    rknn.config(
        mean_values=[123.675, 116.28, 103.53],
        std_values=[58.395, 58.395, 58.395],
        target_platform='rk3588'
    )

    # 使用load_xxx接口加载常用深度学习模型
    rknn.load_pytorch(
        model="./resnet18.pt",
        input_size_list=[[1, 3, 224, 224]]
    )

    # 使用hybrid_quantization_step1接口进行混合量化的第一步
    rknn.hybrid_quantization_step1(
        dataset="dataset.txt", # dataset表示模型量化所需要的数据集
        rknn_batch_size=-1,  # 表示自动调整模型输入batch数量
        proposal=True,  # 设置为True，可以自动产生混合量化的配置建议值
        proposal_dataset_size=1  # 第三步骤所用的图片
    )

    rknn.release()
