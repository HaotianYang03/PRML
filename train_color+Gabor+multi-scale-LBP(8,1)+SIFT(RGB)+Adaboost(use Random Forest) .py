import cv2
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from dataloader import train_loader, test_loader  # 假设已定义数据加载器
from skimage.feature import local_binary_pattern


### 颜色特征提取函数**
def extract_color_features(image_np):
    """
    提取图像的颜色特征（融合RGB/HSV空间的统计量）
    Args:
        image_np: numpy数组，形状 (H, W, C)（C=3为RGB）
    Returns:
        color_feature: 颜色特征向量（一维数组）
    """
    # 转换为HSV颜色空间（更符合人类视觉感知）
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # 1. RGB通道的颜色直方图（全局分布）
    rgb_hist = []
    for channel in range(3):
        hist = cv2.calcHist([image_np], [channel], None, [256], [0, 256])
        rgb_hist.append(hist.flatten() / (image_np.shape[0] * image_np.shape[1]))  # 归一化

    # 2. HSV通道的颜色矩（均值、方差、偏度）
    hsv_moments = []
    for channel in range(3):
        hsv_chan = image_hsv[:, :, channel].astype(np.float32)
        mean = np.mean(hsv_chan)  # 一阶矩（均值）
        var = np.var(hsv_chan)  # 二阶矩（方差）
        skewness = np.mean(((hsv_chan - mean) / (var**0.5 + 1e-8))**3)  # 三阶矩（偏度）
        hsv_moments.extend([mean, var, skewness])

    # 3. 颜色对比度（RGB空间）
    rgb_contrast = np.std(image_np.reshape(-1, 3), axis=0)  # 各通道像素值标准差

    # 融合所有颜色特征（维度：256*3 + 3*3 + 3 = 780）
    color_feature = np.concatenate([
        np.concatenate(rgb_hist),  # RGB直方图（256*3=768维）
        np.array(hsv_moments),  # HSV颜色矩（3*3=9维）
        rgb_contrast  # RGB对比度（3维）
    ])
    return color_feature


### **新增：Gabor纹理特征提取函数**
def extract_gabor_features(image_np):
    """
    提取图像的Gabor纹理特征
    Args:
        image_np: numpy数组，形状 (H, W, C)（C=3为RGB）
    Returns:
        gabor_feature: Gabor纹理特征向量（一维数组）
    """
    # 将图像转为灰度图
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # 定义Gabor滤波器的参数
    gabor_features = []
    num_filters = 8  # 滤波器数量（方向数）
    ksize = 15  # 滤波器大小
    lambdas = [8]  # 波长
    thetas = np.linspace(0, np.pi, num_filters, endpoint=False)  # 方向
    sigmas = [2.0]  # 带宽
    gammas = [0.5]  # 空间纵横比

    # 对每个滤波器计算Gabor特征
    for lam in lambdas:
        for theta in thetas:
            for sigma in sigmas:
                for gamma in gammas:
                    # 创建Gabor滤波器
                    gabor_filter = cv2.getGaborKernel(
                        ksize=(ksize, ksize),
                        sigma=sigma,
                        theta=theta,
                        lambd=lam,
                        gamma=gamma,
                        psi=0,
                        ktype=cv2.CV_32F
                    )
                    # 应用滤波器
                    filtered = cv2.filter2D(gray, cv2.CV_8UC3, gabor_filter)
                    # 计算统计特征
                    mean_val = np.mean(filtered)
                    var_val = np.var(filtered)
                    gabor_features.extend([mean_val, var_val])

    return np.array(gabor_features)


### **步骤1：定义特征提取函数（RGB彩色图像）**
def extract_features_rgb(image_tensor):
    """
    从PyTorch图像张量中提取SIFT、LBP、颜色和Gabor特征（融合三个通道的特征向量）
    Args:
        image_tensor: PyTorch张量，形状 (C, H, W)（C=3为RGB）
    Returns:
        fused_feature: 融合SIFT、LBP、颜色和Gabor特征后的特征向量（一维数组）
    """
    # 转换PyTorch张量为numpy数组（H, W, C）
    image_np = image_tensor.numpy().transpose(1, 2, 0)  # (C, H, W) → (H, W, C)

    # 处理像素值范围（若原数据是0-1的float32，需转换为0-255的uint8）
    if image_np.dtype == np.float32:
        image_np = (image_np * 255).astype(np.uint8)  # 0-1 → 0-255

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 对每个颜色通道分别计算SIFT特征
    sift_features = []
    for channel in range(3):  # 遍历R、G、B三个通道
        single_channel = image_np[:, :, channel]
        kp, des = sift.detectAndCompute(single_channel, None)
        if des is not None:
            # 对描述子进行均值池化（n_kp个128维向量取平均，得到1个128维向量）
            sift_feature = np.mean(des, axis=0)
        else:
            sift_feature = np.zeros(128)
        sift_features.append(sift_feature)

    # 融合三个通道的SIFT特征
    sift_feature_fused = np.concatenate(sift_features)

    # 对每个颜色通道分别计算LBP特征
    lbp_features = []
    for channel in range(3):  # 遍历R、G、B三个通道
        single_channel = image_np[:, :, channel]
        lbp = local_binary_pattern(single_channel, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 58))
        lbp_hist = lbp_hist / lbp_hist.sum()
        lbp_features.append(lbp_hist)

    # 融合三个通道的LBP特征
    lbp_feature_fused = np.concatenate(lbp_features)

    # 提取颜色特征
    color_feature = extract_color_features(image_np)

    # 提取Gabor纹理特征
    gabor_feature = extract_gabor_features(image_np)

    # 融合SIFT、LBP、颜色和Gabor特征
    fused_feature = np.concatenate([sift_feature_fused, lbp_feature_fused, color_feature, gabor_feature])
    return fused_feature


### **步骤2：加载数据并提取特征**
def load_and_extract_features(loader):
    """
    从数据加载器中提取所有样本的SIFT、LBP、颜色和Gabor特征和标签
    Args:
        loader: 数据加载器（train_loader或test_loader）
    Returns:
        features: 所有样本的SIFT、LBP、颜色和Gabor特征（二维数组，形状 (n_samples, feature_dim)）
        labels: 所有样本的标签（一维数组）
    """
    features = []
    labels = []
    for img_batch, label_batch in loader:  # img_batch: (N, C, H, W), label_batch: (N,)
        for i in range(img_batch.shape[0]):  # 遍历batch中的每张图像
            single_img = img_batch[i]  # 单张图像，shape=(C, H, W)
            single_label = label_batch[i]  # 单张图像标签

            # 提取SIFT、LBP、颜色和Gabor特征（RGB彩色图像）
            feature = extract_features_rgb(single_img)
            features.append(feature)
            labels.append(single_label.item())  # 转换标签为标量
    return np.array(features), np.array(labels)


### **步骤3：训练与评估（添加KFold交叉验证）**
if __name__ == "__main__":
    # 提取训练集和测试集的特征（RGB彩色图像）
    print("Extracting training features...")
    X_train, y_train = load_and_extract_features(train_loader)
    print(f"Training features shape: {X_train.shape} (n_samples, feature_dim)")

    print("Extracting test features...")
    X_test, y_test = load_and_extract_features(test_loader)
    print(f"Test features shape: {X_test.shape}")

    # 合并训练集和测试集进行交叉验证
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))

    # 初始化KFold，设置k = 5
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []

    # 进行5折交叉验证
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        # 构建Adaboost分类器（弱分类器使用随机森林）
        # 标准化：Adaboost对特征尺度敏感，需将特征缩放到均值0、方差1
        clf = make_pipeline(
            StandardScaler(),
            AdaBoostClassifier(
                estimator=RandomForestClassifier(n_estimators=50, random_state=42),
                n_estimators=10,
                learning_rate=0.1,
                random_state=42
            )
        )

        # 训练Adaboost
        print(f"Training on fold {fold + 1}...")
        clf.fit(X_train_fold, y_train_fold)

        # 评估验证集准确率
        y_val_pred = clf.predict(X_val_fold)
        fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
        fold_accuracies.append(fold_accuracy)
        # 按要求保留三位小数
        print(f"Validation accuracy on fold {fold + 1}: {fold_accuracy:.3f}")

    # 计算平均准确率和标准差
    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)

    # 输出最终结果，保留三位小数
    print("\nCross-validation results:")
    print(f"5-fold Accuracies: {[f'{acc:.3f}' for acc in fold_accuracies]}")
    print(f"Average Accuracy: {avg_accuracy:.3f}")
    print(f"Standard Deviation: {std_accuracy:.3f}")

    # 最后，使用完整的训练集训练模型并在测试集上评估
    print("\nTraining on full training set and evaluating on test set...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.3f}")