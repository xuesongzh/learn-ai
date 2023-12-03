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

    # 使用build接口构建RKNN模型
    rknn.build(
        do_quantization=True,
        dataset='dataset.txt'
    )

    # 使用expoer_rknn接口导出RKNN模型
    rknn.export_rknn(
        export_path="./resnet18.rknn"
    )

    # 使用accuracy_analysis 接口进行模型量化精度分析
    rknn.accuracy_analysis(
        inputs=["space_shuttle_224.jpg"],  # inputs用来表示进行推理的图像
        output_dir="snapshot",  # output_dir表示精度分析的输出目录
        target="rk3588",  # target表示目标硬件平台
        device_id=None,  # device_id表示设备的编号
    )
    # 使用release接口释放RKNN模型
    rknn.release()
