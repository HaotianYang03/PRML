import argparse
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from torchvision.transforms import Grayscale
from dataloader import PollenDataset, image_paths, class_names, labels, transform


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
            'gabor': self._extract_gabor
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
            }
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
    def __getitem__(self, index):
        img_tensor, label = super().__getitem__(index)
        img_np = img_tensor.numpy().transpose(1, 2, 0) * 255
        img_np = img_np.astype(np.uint8)

        features = {}
        for ch in range(3):  # 处理三个通道
            channel_img = img_np[:, :, ch]
            for feat_name in self.selected_features:
                feat = self.feature_extractors[feat_name](channel_img)
                features.setdefault(feat_name, []).append(feat)

        # 合并通道特征
        combined = {
            feat_name: np.concatenate(features[feat_name]).astype(np.float32)
            for feat_name in self.selected_features
        }
        return combined, label


def parse_args():
    parser = argparse.ArgumentParser(description='特征提取流水线')
    parser.add_argument('--features', type=str, default='hog,lbp,glcm,gabor',
                        help='要提取的特征列表，用逗号分隔 (默认:全部)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    selected_features = [f.strip().lower() for f in args.features.split(',')]

    # 验证特征名称
    valid_features = {'hog', 'lbp', 'glcm', 'gabor'}
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
        np.save(f'{feat}_features.npy', np.array(feature_data[feat]))
        print(f"{feat.upper()}特征已保存，维度: {np.array(feature_data[feat]).shape}")

# import argparse
# import numpy as np
# from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
# from skimage.filters import gabor
# from scipy import ndimage
# from skimage.transform import resize
# from torchvision.transforms import Grayscale
# from dataloader import PollenDataset, image_paths, class_names, labels, transform
# import pywt
#
#
# class FeatureExtractor(PollenDataset):
#     def __init__(self, selected_features, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.grayscale = Grayscale(num_output_channels=1)
#         self.selected_features = selected_features
#
#         # 特征提取器配置
#         self.feature_extractors = {
#             'lbp': self._extract_lbp,
#             'glcm': self._extract_glcm,
#             'gabor': self._extract_gabor,
#             'fos': self._extract_fos
#         }
#
#         # 参数配置
#         self.feature_params = {
#             'lbp': {
#                 'P': 24,
#                 'R': 3.0,
#                 'method': 'uniform',
#                 'blocks': (4, 4),
#                 'bins': 26
#             },
#             'glcm': {
#                 'distances': [1],
#                 'angles': [0, np.pi / 4],
#                 'properties': ['contrast', 'homogeneity'],
#                 'blocks': (3, 3)
#             },
#             'gabor': {
#                 'frequencies': [0.1, 0.3],
#                 'thetas': np.linspace(0, np.pi, 3),
#                 'sigma': 1.5
#             },
#             'fos': {
#                 'properties': ['mean', 'std', 'skew', 'kurtosis', 'entropy']
#             }
#         }
#
#     # 特征提取方法 --------------------------------------------------
#     def _extract_fos(self, img):
#         """一阶统计量"""
#         img_float = img.astype(np.float32) / 255.0
#         features = []
#
#         if 'mean' in self.feature_params['fos']['properties']:
#             features.append(np.mean(img_float))
#         if 'std' in self.feature_params['fos']['properties']:
#             features.append(np.std(img_float))
#         if 'skew' in self.feature_params['fos']['properties']:
#             features.append(ndimage.sobel(img_float).std())
#         if 'kurtosis' in self.feature_params['fos']['properties']:
#             features.append(ndimage.gaussian_filter(img_float, sigma=1).std())
#         if 'entropy' in self.feature_params['fos']['properties']:
#             hist = np.histogram(img, bins=256, range=(0, 255))[0]
#             hist = hist / hist.sum()
#             entropy = -np.sum(hist * np.log2(hist + 1e-10))
#             features.append(entropy)
#
#         return np.array(features)
#
#     def _extract_lbp(self, img):
#         lbp = local_binary_pattern(img,
#                                    self.feature_params['lbp']['P'],
#                                    self.feature_params['lbp']['R'],
#                                    self.feature_params['lbp']['method'])
#         blocks = self.feature_params['lbp']['blocks']
#         block_h, block_w = img.shape[0] // blocks[0], img.shape[1] // blocks[1]
#         hist = []
#         for i in range(blocks[0]):
#             for j in range(blocks[1]):
#                 block = lbp[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
#                 hist_block, _ = np.histogram(block,
#                                              bins=self.feature_params['lbp']['bins'],
#                                              range=(0, 26))
#                 hist.extend(hist_block)
#         return np.array(hist)
#
#     def _extract_glcm(self, img):
#         glcm_features = []
#         blocks = self.feature_params['glcm']['blocks']
#         block_h, block_w = img.shape[0] // blocks[0], img.shape[1] // blocks[1]
#         for i in range(blocks[0]):
#             for j in range(blocks[1]):
#                 block = img[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
#                 glcm = graycomatrix(block.astype(np.uint8),
#                                     distances=self.feature_params['glcm']['distances'],
#                                     angles=self.feature_params['glcm']['angles'],
#                                     levels=256,
#                                     symmetric=True)
#                 features = []
#                 for prop in self.feature_params['glcm']['properties']:
#                     features.extend(graycoprops(glcm, prop).ravel())
#                 glcm_features.extend(features)
#         return np.array(glcm_features)
#
#     def _extract_gabor(self, img):
#         features = []
#         for freq in self.feature_params['gabor']['frequencies']:
#             for theta in self.feature_params['gabor']['thetas']:
#                 real, imag = gabor(img, frequency=freq,
#                                    theta=theta,
#                                    sigma_x=self.feature_params['gabor']['sigma'])
#                 features.extend([real.mean(), real.std(), imag.mean(), imag.std()])
#         return np.array(features)
#
#     # 投影预处理 --------------------------------------------------
#     def _preprocess_projections(self, img_np):
#         """三投影预处理"""
#         processed_projections = []
#         for proj_idx, proj_type in enumerate(['STD', 'MIN', 'EXT']):
#             proj_img = img_np[:, :, proj_idx]
#
#             # 投影特定处理
#             if proj_type == 'STD':
#                 blurred = ndimage.gaussian_filter(proj_img, sigma=1)
#                 proj_img = np.clip(proj_img + (proj_img - blurred), 0, 255)
#             elif proj_type == 'MIN':
#                 p2, p98 = np.percentile(proj_img, (2, 98))
#                 if p2 != p98:
#                     proj_img = (proj_img - p2) * (255 / (p98 - p2))
#             elif proj_type == 'EXT':
#                 dx = ndimage.sobel(proj_img, axis=0)
#                 dy = ndimage.sobel(proj_img, axis=1)
#                 mag = np.hypot(dx, dy)
#                 proj_img = np.clip(proj_img + mag * 0.3, 0, 255)
#
#             processed_projections.append(proj_img)
#
#         return processed_projections
#
#     def __getitem__(self, index):
#         img_tensor, label = super().__getitem__(index)
#         img_np = img_tensor.numpy().transpose(1, 2, 0) * 255
#         img_np = img_np.astype(np.uint8)
#
#         # 存储所有特征的字典
#         all_features = {}
#
#         # 1. 处理投影图像特征（STD, MIN, EXT）
#         processed_projections = self._preprocess_projections(img_np)
#
#         # 提取投影特征
#         projection_features = {}
#         for proj_idx, proj_img in enumerate(processed_projections):
#             proj_feats = {}
#             for feat_name in self.selected_features:
#                 if feat_name in self.feature_extractors:
#                     feat = self.feature_extractors[feat_name](proj_img)
#                     proj_feats[feat_name] = feat
#
#             # 按投影保存特征
#             for feat_name, feat_value in proj_feats.items():
#                 key = f"{feat_name}_proj{proj_idx}"
#                 projection_features[key] = feat_value
#
#         # 合并投影特征
#         for feat_name in self.selected_features:
#             if feat_name in self.feature_extractors:
#                 all_features[feat_name] = np.concatenate([
#                     projection_features[f"{feat_name}_proj{proj_idx}"]
#                     for proj_idx in range(3)
#                 ])
#
#         # 2. 处理RGB通道特征
#         rgb_features = {}
#         for ch in range(3):  # 处理三个通道
#             channel_img = img_np[:, :, ch]
#             for feat_name in ['gabor', 'glcm', 'fos']:
#                 if feat_name in self.feature_extractors:
#                     feat = self.feature_extractors[feat_name](channel_img)
#                     rgb_features.setdefault(feat_name, []).append(feat)
#
#         # 合并RGB通道特征
#         for feat_name in ['gabor', 'glcm', 'fos']:
#             if feat_name in rgb_features:
#                 rgb_key = f"{feat_name}_rgb"
#                 all_features[rgb_key] = np.concatenate(rgb_features[feat_name])
#
#         # 3. 按固定顺序组合所有特征
#         feature_order = [
#             'lbp', 'gabor', 'glcm', 'fos',  # 投影特征
#             'gabor_rgb', 'glcm_rgb', 'fos_rgb'  # RGB特征
#         ]
#
#         # 收集所有特征向量
#         feature_vector = []
#         for feat_name in feature_order:
#             if feat_name in all_features:
#                 feature_vector.append(all_features[feat_name])
#
#         # 合并所有特征向量
#         combined_feature = np.concatenate(feature_vector)
#
#         return combined_feature, label
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='特征提取流水线')
#     parser.add_argument('--features', type=str, default='lbp,gabor,glcm,fos',
#                         help='要提取的投影特征列表，用逗号分隔 (默认: lbp,gabor,glcm,fos)')
#     return parser.parse_args()
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     selected_features = [f.strip().lower() for f in args.features.split(',')]
#
#     # 验证特征名称
#     valid_features = {'lbp', 'gabor', 'glcm', 'fos'}
#     invalid = set(selected_features) - valid_features
#     if invalid:
#         raise ValueError(f"无效特征名称: {invalid}。可用特征: {valid_features}")
#
#     # 初始化数据集
#     feature_dataset = FeatureExtractor(
#         selected_features,
#         image_paths,
#         labels,
#         transform=transform
#     )
#
#     # 存储所有特征
#     all_features = []
#     labels_list = []
#
#     # 批量处理
#     for i in range(len(feature_dataset)):
#         if i % 10 == 0:
#             print(f"Processing sample {i + 1}/{len(feature_dataset)}")
#         features, label = feature_dataset[i]
#         all_features.append(features)
#         labels_list.append(label)
#
#     # 保存文件
#     all_features = np.array(all_features)
#     labels_list = np.array(labels_list)
#
#     np.save('feature.npy', all_features)
#     np.save('labels.npy', labels_list)
#
#     print(f"所有特征已保存到 feature.npy, 维度: {all_features.shape}")
#     print(f"标签已保存到 labels.npy, 维度: {labels_list.shape}")