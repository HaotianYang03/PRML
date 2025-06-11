
import argparse
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from torchvision.transforms import Grayscale
from dataloader import PollenDataset, image_paths, class_names, labels, transform
# import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from skimage.util import view_as_blocks
from scipy import ndimage
from scipy.fftpack import dct
from skimage.transform import resize
from torchvision.transforms import Grayscale
from dataloader import PollenDataset, image_paths, class_names, labels, transform
import pywt

class FeatureExtractor(PollenDataset):
    def __init__(self, selected_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grayscale = Grayscale(num_output_channels=1)
        self.selected_features = selected_features

        # 特征提取器配置
        self.feature_extractors = {
            'hog': self._extract_hog,
            'lbp': self._extract_lbp,
            'glcm': self._extract_glcm,
            'gabor': self._extract_gabor,
            'fos': self._extract_fos,
            'wavelet': self._extract_wavelet,
            'hu_moments': self._extract_hu_moments,
            'ede': self._extract_ede
        }

        # 参数配置（可根据需要扩展）
        self.feature_params = {
            'hog': {
                'orientations': 9,
                'pixels_per_cell': (32, 32),
                'cells_per_block': (3, 3),
                'block_norm': 'L2-Hys',
                'transform_sqrt': True,
                'feature_vector': True,
                'channel_axis': None
            },
            'lbp': {
                'P': 24,
                'R': 3.0,
                'method': 'uniform',
                'blocks': (4, 4),
                'bins': 26
            },
            'glcm': {
                'distances': [1],
                'angles': [0, np.pi / 4],
                'properties': ['contrast', 'homogeneity'],
                'blocks': (3, 3)
            },
            'gabor': {
                'frequencies': [0.1, 0.3],
                'thetas': np.linspace(0, np.pi, 3),
                'sigma': 1.5
            },
            'fos': {
                'properties': ['mean', 'std', 'skew', 'kurtosis', 'entropy']
            },
            'wavelet': {
                'wavelet': 'db4',
                'levels': 3,
                'resize_shape': (128, 128)
            },
            'hu_moments': {
                'threshold': 0.5  # 二值化阈值
            },
            'ede': {
                'threshold': 0.5
            }
        }

        # 新增特征提取方法 --------------------------------------------------
    def _extract_fos(self, img):
        """一阶统计量"""
        img_float = img.astype(np.float32) / 255.0
        features = []

        if 'mean' in self.feature_params['fos']['properties']:
            features.append(np.mean(img_float))
        if 'std' in self.feature_params['fos']['properties']:
            features.append(np.std(img_float))
        if 'skew' in self.feature_params['fos']['properties']:
            features.append(ndimage.sobel(img_float).std())
        if 'kurtosis' in self.feature_params['fos']['properties']:
            features.append(ndimage.gaussian_filter(img_float, sigma=1).std())
        if 'entropy' in self.feature_params['fos']['properties']:
            hist = np.histogram(img, bins=256, range=(0, 255))[0]
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            features.append(entropy)

        return np.array(features)

    def _extract_wavelet(self, img):
        """小波变换特征"""
        # 预处理
        resized_img = resize(img, self.feature_params['wavelet']['resize_shape'])

        # 多级小波分解
        coeffs = pywt.wavedec2(resized_img,
                               self.feature_params['wavelet']['wavelet'],
                               level=self.feature_params['wavelet']['levels'])

        features = []
        # 提取各子带统计量
        for i, coeff in enumerate(coeffs):
            if i == 0:  # 近似分量
                cA = coeff
                features.extend([cA.mean(), cA.std(), np.median(cA)])
            else:  # 细节分量
                (cH, cV, cD) = coeff
                for band in [cH, cV, cD]:
                    features.extend([band.mean(), band.std(), np.median(band)])

        return np.array(features[:30])  # 保持特征维度一致

    import numpy as np
    from scipy import ndimage
    from scipy.spatial import ConvexHull

    def _extract_hu_moments(self, img):
        """Hu矩特征替代实现"""
        # 二值化处理
        threshold = int(255 * self.feature_params['hu_moments']['threshold'])
        binary_img = np.where(img >= threshold, 1.0, 0.0).astype(np.float32)

        # 计算空间矩
        m = self._raw_image_moments(binary_img)

        # 计算Hu矩
        hu = np.zeros(7, dtype=np.float32)
        if m['m00'] != 0:
            # 计算归一化中心矩
            nu20 = m['mu20'] / m['m00'] ** 2.5
            nu02 = m['mu02'] / m['m00'] ** 2.5
            nu11 = m['mu11'] / m['m00'] ** 2.5
            nu30 = m['mu30'] / m['m00'] ** 3.5
            nu12 = m['mu12'] / m['m00'] ** 3.5
            nu21 = m['mu21'] / m['m00'] ** 3.5
            nu03 = m['mu03'] / m['m00'] ** 3.5

            # Hu's 7 invariant moments
            hu[0] = nu20 + nu02
            hu[1] = (nu20 - nu02) ** 2 + 4 * nu11 ** 2
            hu[2] = (nu30 - 3 * nu12) ** 2 + (3 * nu21 - nu03) ** 2
            hu[3] = (nu30 + nu12) ** 2 + (nu21 + nu03) ** 2
            hu[4] = (nu30 - 3 * nu12) * (nu30 + nu12) * ((nu30 + nu12) ** 2 - 3 * (nu21 + nu03) ** 2) + \
                    (3 * nu21 - nu03) * (nu21 + nu03) * (3 * (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2)
            hu[5] = (nu20 - nu02) * ((nu30 + nu12) ** 2 - (nu21 + nu03) ** 2) + \
                    4 * nu11 * (nu30 + nu12) * (nu21 + nu03)
            hu[6] = (3 * nu21 - nu03) * (nu30 + nu12) * ((nu30 + nu12) ** 2 - 3 * (nu21 + nu03) ** 2) - \
                    (nu30 - 3 * nu12) * (nu21 + nu03) * (3 * (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2)

        # 对数变换
        return -np.sign(hu) * np.log(np.abs(hu) + 1e-10)

    def _raw_image_moments(self, img):
        """计算原始图像矩"""
        y, x = np.mgrid[:img.shape[0], :img.shape[1]]
        m00 = np.sum(img)
        m10 = np.sum(x * img)
        m01 = np.sum(y * img)
        cx = m10 / (m00 + 1e-10)
        cy = m01 / (m00 + 1e-10)

        # 中心矩
        mu20 = np.sum((x - cx) ** 2 * img)
        mu02 = np.sum((y - cy) ** 2 * img)
        mu11 = np.sum((x - cx) * (y - cy) * img)
        mu30 = np.sum((x - cx) ** 3 * img)
        mu03 = np.sum((y - cy) ** 3 * img)
        mu12 = np.sum((x - cx) * (y - cy) ** 2 * img)
        mu21 = np.sum((x - cx) ** 2 * (y - cy) * img)

        return {
            'm00': m00, 'm10': m10, 'm01': m01,
            'mu20': mu20, 'mu02': mu02, 'mu11': mu11,
            'mu30': mu30, 'mu03': mu03, 'mu12': mu12, 'mu21': mu21
        }

    def _extract_ede(self, img):
        """EDE形状特征替代实现"""
        # 二值化
        threshold = int(255 * self.feature_params['ede']['threshold'])
        binary = np.where(img >= threshold, 1, 0).astype(np.uint8)

        # 寻找轮廓
        contours = self._find_contours(binary)
        if not contours:
            return np.zeros(3)

        # 选择最大轮廓
        largest_cnt = max(contours, key=lambda c: c['area'])

        # 计算轮廓矩
        m = self._contour_moments(largest_cnt['points'])

        # 计算EDE
        extension = (m['mu20'] + m['mu02']) / (m['m00'] ** 2 + 1e-10)
        dispersion = (m['mu20'] - m['mu02']) ** 2 + 4 * m['mu11'] ** 2
        elongation = np.sqrt((m['mu20'] + m['mu02'] + np.sqrt(dispersion)) /
                             (m['mu20'] + m['mu02'] - np.sqrt(dispersion) + 1e-10))

        return np.array([extension, dispersion, elongation])

    def _find_contours(self, binary):
        """轮廓查找算法"""
        # 使用边缘检测
        edges = np.abs(ndimage.sobel(binary, axis=0)) + np.abs(ndimage.sobel(binary, axis=1))
        edges = (edges > 0).astype(np.uint8)

        # 连通区域标记
        labeled, num_features = ndimage.label(edges)

        # 提取轮廓
        contours = []
        for i in range(1, num_features + 1):
            points = np.argwhere(labeled == i)
            if len(points) < 5:  # 过滤小轮廓
                continue
            contours.append({
                'points': points[:, [1, 0]],  # 转换为(x,y)格式
                'area': len(points)
            })
        return contours

    def _contour_moments(self, points):
        """计算轮廓矩"""
        x, y = points[:, 0], points[:, 1]
        m00 = len(points)
        m10 = np.sum(x)
        m01 = np.sum(y)
        cx = m10 / m00
        cy = m01 / m00

        mu20 = np.sum((x - cx) ** 2)
        mu02 = np.sum((y - cy) ** 2)
        mu11 = np.sum((x - cx) * (y - cy))

        return {
            'm00': m00, 'mu20': mu20, 'mu02': mu02, 'mu11': mu11
        }
    # 特征提取方法 --------------------------------------------------
    def _extract_hog(self, img):
        return hog(img, **self.feature_params['hog'])

    def _extract_lbp(self, img):
        lbp = local_binary_pattern(img, **{k: v for k, v in self.feature_params['lbp'].items()
                                           if k in ['P', 'R', 'method']})
        blocks = self.feature_params['lbp']['blocks']
        block_h, block_w = img.shape[0] // blocks[0], img.shape[1] // blocks[1]
        hist = []
        for i in range(blocks[0]):
            for j in range(blocks[1]):
                block = lbp[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
                hist_block, _ = np.histogram(block,
                                             bins=self.feature_params['lbp']['bins'],
                                             range=(0, 26))
                hist.extend(hist_block)
        return np.array(hist)

    def _extract_glcm(self, img):
        glcm_features = []
        blocks = self.feature_params['glcm']['blocks']
        block_h, block_w = img.shape[0] // blocks[0], img.shape[1] // blocks[1]
        for i in range(blocks[0]):
            for j in range(blocks[1]):
                block = img[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
                glcm = graycomatrix(block.astype(np.uint8),
                                    distances=self.feature_params['glcm']['distances'],
                                    angles=self.feature_params['glcm']['angles'],
                                    levels=256,
                                    symmetric=True)
                features = []
                for prop in self.feature_params['glcm']['properties']:
                    features.extend(graycoprops(glcm, prop).ravel())
                glcm_features.extend(features)
        return np.array(glcm_features)

    def _extract_gabor(self, img):
        features = []
        for freq in self.feature_params['gabor']['frequencies']:
            for theta in self.feature_params['gabor']['thetas']:
                real, imag = gabor(img, frequency=freq,
                                   theta=theta,
                                   sigma_x=self.feature_params['gabor']['sigma'])
                features.extend([real.mean(), real.std(), imag.mean(), imag.std()])
        return np.array(features)

    # 数据管道 --------------------------------------------------
    def _preprocess_projections(self, img_np):
        """
        新增方法: 三投影预处理
        最小修改点1: 在这里添加投影特定处理
        """
        processed_projections = []
        for proj_idx, proj_type in enumerate(['STD', 'MIN', 'EXT']):
            proj_img = img_np[:, :, proj_idx]

            # 投影特定处理 - 最小化修改的关键点
            if proj_type == 'STD':
                # STD投影: 简单纹理增强
                blurred = ndimage.gaussian_filter(proj_img, sigma=1)
                proj_img = np.clip(proj_img + (proj_img - blurred), 0, 255)
            elif proj_type == 'MIN':
                # MIN投影: 对比度增强
                p2, p98 = np.percentile(proj_img, (2, 98))
                if p2 != p98:
                    proj_img = (proj_img - p2) * (255 / (p98 - p2))
            elif proj_type == 'EXT':
                # EXT投影: 边缘增强
                dx = ndimage.sobel(proj_img, axis=0)
                dy = ndimage.sobel(proj_img, axis=1)
                mag = np.hypot(dx, dy)
                proj_img = np.clip(proj_img + mag * 0.3, 0, 255)

            processed_projections.append(proj_img)

        return processed_projections

    def __getitem__(self, index):
        """
        最小修改点2: 在此函数中增加投影处理层
        """
        img_tensor, label = super().__getitem__(index)
        img_np = img_tensor.numpy().transpose(1, 2, 0) * 255
        img_np = img_np.astype(np.uint8)

        # 新增三投影处理 - 主要修改点
        processed_projections = self._preprocess_projections(img_np)

        features = {}
        for proj_idx, proj_img in enumerate(processed_projections):
            proj_features = {}
            for feat_name in self.selected_features:
                if feat_name in self.feature_extractors:
                    # 使用现有的特征提取方法 - 核心逻辑不变
                    feat = self.feature_extractors[feat_name](proj_img)
                    proj_features[feat_name] = feat

            # 按投影添加前缀保存特征 - 最小修改点3
            for feat_name, feat_value in proj_features.items():
                key = f"{feat_name}_{proj_idx}"
                features[key] = feat_value

        # 合并特征向量 - 最小修改点4
        combined = {}
        for feat_name in self.selected_features:
            # 收集所有投影的同类型特征
            all_proj_feats = []
            for proj_idx in range(3):  # 3个投影
                key = f"{feat_name}_{proj_idx}"
                if key in features:
                    if not isinstance(features[key], np.ndarray):
                        all_proj_feats.append(np.array([features[key]]))
                    else:
                        all_proj_feats.append(features[key])

            # 连接三个投影的特征向量
            if all_proj_feats:
                combined[feat_name] = np.concatenate(all_proj_feats).astype(np.float32)
            else:
                combined[feat_name] = np.zeros(10).astype(np.float32)

        return combined, label


def parse_args():
    parser = argparse.ArgumentParser(description='特征提取流水线')
    parser.add_argument('--features', type=str, default='hog,lbp,glcm,gabor,fos,wavelet,hu_moments,ede',
                        help='要提取的特征列表，用逗号分隔 (默认:全部)')##hog,lbp,glcm,gabor,
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    selected_features = [f.strip().lower() for f in args.features.split(',')]

    # 验证特征名称
    valid_features = {'hog', 'lbp', 'glcm', 'gabor','fos','wavelet','hu_moments','ede'}
    invalid = set(selected_features) - valid_features
    if invalid:
        raise ValueError(f"无效特征名称: {invalid}。可用特征: {valid_features}")

    # 初始化数据集
    feature_dataset = FeatureExtractor(selected_features,
                                       image_paths,
                                       labels,
                                       transform=transform)

    # 初始化存储
    feature_data = {feat: [] for feat in selected_features}
    labels = []

    # 批量处理
    for i in range(len(feature_dataset)):
        if i % 10 == 0:
            print(i)
        features, label = feature_dataset[i]
        for feat in selected_features:
            feature_data[feat].append(features[feat])
        labels.append(label)

    # 保存文件
    np.save('labels.npy', np.array(labels))
    for feat in selected_features:
        np.save(f'{feat}_features3.npy', np.array(feature_data[feat]))
        print(f"{feat.upper()}特征已保存，维度: {np.array(feature_data[feat]).shape}")