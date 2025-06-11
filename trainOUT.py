# import numpy as np
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import cross_val_score
#
#
# def load_features():
#     """从文件加载特征和标签"""
#     # 加载三个特征文件
#     hog_features = np.load('hog_features.npy')
#     lbp_features = np.load('lbp_features.npy')#80acc
#     glcm_features = np.load('glcm_features.npy')# 71acc
#     gabor_features = np.load('gabor_features.npy')# 83acc
#     labels = np.load('labels.npy')
#
#     # 验证数据一致性
#     assert len(hog_features) == len(lbp_features) == len(labels), "特征与标签数量不匹配"
#
#     # 合并HOG和LBP特征hog_features, lbp_features
#     combined_features = np.concatenate([gabor_features], axis=1)
#
#     # # 特征维度验证（根据原特征提取参数）
#     # assert hog_features.shape[1] == 8748, "HOG特征维度异常"  # 3通道各3600维
#     # assert lbp_features.shape[1] == 1248, "LBP特征维度异常"  # 3通道各416维
#     print(f"特征加载完成，总样本数: {len(labels)}")
#
#     return combined_features, labels
#
#
# def train_traditional_model():
#     # 加载数据
#     X, y = load_features()
#     print(f"特征矩阵形状: {X.shape}, 标签数量: {len(y)}")
#
#     # 构建分类流水线
#     model = make_pipeline(
#         StandardScaler(),
#         SVC(C=4, kernel='rbf', random_state=42)
#     )
#
#     # 交叉验证评估
#     scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
#     print(f"交叉验证准确率: {scores.mean():.3f} ± {scores.std():.3f}")
#
#     # 全数据训练并保存模型
#     model.fit(X, y)
#     import joblib
#     joblib.dump(model, 'hog_lbp_svm.pkl')
#     print("模型训练完成并已保存")
#
#
# if __name__ == '__main__':
#     train_traditional_model()

#
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import cross_val_score
# import itertools
#
# def load_features():
#     """从文件加载特征和标签"""
#     # 加载四个特征文件
#     hog_features = np.load('hog_features.npy')
#     lbp_features = np.load('lbp_features.npy')
#     glcm_features = np.load('glcm_features.npy')
#     gabor_features = np.load('gabor_features.npy')
#     fos_features = np.load('fos_features.npy')
#     wavelet_features = np.load('wavelet_features.npy')
#     hu_moments_features = np.load('hu_moments_features.npy')
#     ede_features = np.load('ede_features.npy')
#     labels = np.load('labels.npy')
#
#     # 验证数据一致性
#     assert len(hog_features) == len(lbp_features) == len(glcm_features) == len(gabor_features) == len(labels), "特征与标签数量不匹配"
#     print(f"特征加载完成，总样本数: {len(labels)}")
#
#     return hog_features, lbp_features, glcm_features, gabor_features, labels
#
# def evaluate_combinations():
#     # 加载数据
#     hog, lbp, glcm, gabor, y = load_features()
#     feature_dict = {
#         'hog': hog,
#         'lbp': lbp,
#         'glcm': glcm,
#         'gabor': gabor
#     }
#
#     # 生成所有非空组合（共15种）
#     all_combinations = []
#     for r in range(1, 5):
#         combos = itertools.combinations(feature_dict.keys(), r)
#         all_combinations.extend(combos)
#
#     # 遍历每个组合并进行评估
#     for combo in all_combinations:
#         # 合并特征
#         X = np.concatenate([feature_dict[name] for name in combo], axis=1)
#         print(f"\n当前组合: {combo}, 特征维度: {X.shape[1]}")
#
#         # 构建模型流水线
#         model = make_pipeline(
#             StandardScaler(),
#             SVC(C=4, kernel='rbf', random_state=42)
#         )
#
#         # 交叉验证
#         try:
#             scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
#             avg_score = np.mean(scores)
#             std_score = np.std(scores)
#             print(f"交叉验证准确率: {avg_score:.3f} ± {std_score:.3f}")
#         except Exception as e:
#             print(f"评估组合 {combo} 时出错: {str(e)}")
#
# if __name__ == '__main__':
#     evaluate_combinations()

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import itertools
import time
import json

from sklearn.decomposition import PCA  # 添加PCA导入


def load_features():
    """加载全部8个特征和标签"""
    features = {
        'feature': np.load('feature.npy'),
        'hog': np.load('hog_features.npy'),
        'lbp': np.load('lbp_features.npy'),
        'glcm': np.load('glcm_features.npy'),
        'gabor': np.load('gabor_features.npy'),
        'fos': np.load('fos_features.npy'),
        'wavelet': np.load('wavelet_features.npy'),
        'hu_moments': np.load('hu_moments_features.npy'),
        'ede': np.load('ede_features.npy'),
        'hog3': np.load('hog_features3.npy'),
        'lbp3': np.load('lbp_features3.npy'),
        'glcm3': np.load('glcm_features3.npy'),
        'gabor3': np.load('gabor_features3.npy'),
        'fos3': np.load('fos_features3.npy'),
        'wavelet3': np.load('wavelet_features3.npy'),
        'hu_moments3': np.load('hu_moments_features3.npy'),
        'ede3': np.load('ede_features3.npy')
    }
    labels = np.load('labels.npy')

    # ==== 在此处为hog特征添加PCA降维 ====
    if 'hog' in features:
        hog_data = features['hog']
        orig_dim = hog_data.shape[1]

        # 根据原始维度决定降维策略
        if orig_dim > 1000:
            # 原始维度 > 1000: 降维至1000维
            pca = PCA(n_components=1000, random_state=42)
            hog_reduced = pca.fit_transform(hog_data)
            features['hog'] = hog_reduced
            print(f"HOG特征已降维: {orig_dim} → 1000维 (保留方差: {pca.explained_variance_ratio_.sum():.2%})")
        else:
            # 原始维度 ≤ 1000: 使用95%解释方差自动降维
            pca = PCA(n_components=0.95, random_state=42)
            hog_reduced = pca.fit_transform(hog_data)
            features['hog'] = hog_reduced
            print(
                f"HOG特征自动降维: {orig_dim} → {hog_reduced.shape[1]}维 (保留方差: {pca.explained_variance_ratio_.sum():.2%})")
    # ===== PCA结束 =====

    # 数据一致性验证
    sample_counts = {k: v.shape[0] for k, v in features.items()}
    assert all(c == labels.shape[0] for c in sample_counts.values()), "特征样本数不一致"
    print("特征维度详情:")
    for name, data in features.items():
        print(f"{name:12} => 样本数: {data.shape[0]:<5} 特征维度: {data.shape[1]}")

    return features, labels


def load_featuresORI():
    """加载全部8个特征和标签"""
    features = {
        # 'feature': np.load('feature.npy'),
        # 'feature1': np.load('feature1.npy'),
        'hog': np.load('hog_features.npy'),
        'lbp': np.load('lbp_features.npy'),
        'glcm': np.load('glcm_features.npy'),
        'gabor': np.load('gabor_features.npy'),
        'fos': np.load('fos_features.npy'),
        'wavelet': np.load('wavelet_features.npy'),
        'hu_moments': np.load('hu_moments_features.npy'),
        'ede': np.load('ede_features.npy'),
        'hog3': np.load('hog_features3.npy'),
        'lbp3': np.load('lbp_features3.npy'),
        'glcm3': np.load('glcm_features3.npy'),
        'gabor3': np.load('gabor_features3.npy'),
        'fos3': np.load('fos_features3.npy'),
        'wavelet3': np.load('wavelet_features3.npy'),
        'hu_moments3': np.load('hu_moments_features3.npy'),
        'ede3': np.load('ede_features3.npy')
    }
    labels = np.load('labels.npy')

    # 数据一致性验证
    sample_counts = {k: v.shape[0] for k, v in features.items()}
    assert all(c == labels.shape[0] for c in sample_counts.values()), "特征样本数不一致"
    print("特征维度详情:")
    for name, data in features.items():
        print(f"{name:12} => 样本数: {data.shape[0]:<5} 特征维度: {data.shape[1]}")

    return features, labels


def evaluate_combinations(max_features=4):
    """评估特征组合，max_features限制最大组合特征数"""
    features, y = load_features()
    results = []
    ITERfeatures = {
        # 'feature1': np.load('feature1.npy'),
        # 'hog': np.load('hog_features.npy'),
        # 'lbp': np.load('lbp_features.npy'),
        # 'glcm': np.load('glcm_features.npy'),
        # 'gabor': np.load('gabor_features.npy'),
        # 'fos': np.load('fos_features.npy'),
        # 'wavelet': np.load('wavelet_features.npy'),
        # 'hu_moments': np.load('hu_moments_features.npy'),
        # 'ede': np.load('ede_features.npy'),
        # # 'hog3': np.load('hog_features3.npy'),
        # 'lbp3': np.load('lbp_features3.npy'),
        # 'glcm3': np.load('glcm_features3.npy'),
        # 'gabor3': np.load('gabor_features3.npy'),
        'fos3': np.load('fos_features3.npy'),
        # 'wavelet3': np.load('wavelet_features3.npy'),
        # 'hu_moments3': np.load('hu_moments_features3.npy'),
        # 'ede3': np.load('ede_features3.npy')
    }
    # 生成所有有效组合
    for r in range(1, max_features + 1):
        for combo in itertools.combinations(ITERfeatures.keys(), r):
            start_time = time.time()
            FIXED_features = {
                # 'feature': np.load('feature1.npy'),
                'glcm': np.load('glcm_features.npy'),
                'gabor': np.load('gabor_features.npy'),
                'lbp3': np.load('lbp_features3.npy'),
                'gabor3': np.load('gabor_features3.npy'),
                'glcm3': np.load('glcm_features3.npy'),
                'fos': np.load('fos_features.npy'),
                # 'ede': np.load('ede_features.npy'),
            }
            combo = combo+tuple(FIXED_features.keys())
            try:
                # 合并特征
                X = np.concatenate([features[name] for name in combo], axis=1)

                # 构建模型
                model = make_pipeline(
                    StandardScaler(),
                    SVC(C=4, kernel='rbf', random_state=42, cache_size=1000)
                )

                # 交叉验证
                scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)

                # 记录结果
                result = {
                    'combination': combo,
                    'dimension': X.shape[1],
                    'mean_accuracy': round(np.mean(scores), 4),
                    'std_accuracy': round(np.std(scores), 4),
                    'time_sec': round(time.time() - start_time, 1)
                }
                results.append(result)

                # 实时输出
                print(f"{result['combination']} => "
                      f"Acc: {result['mean_accuracy']:.3f}±{result['std_accuracy']:.3f} "
                      f"({result['dimension']}D) "
                      f"Time: {result['time_sec']}s")

            except Exception as e:
                print(f"组合 {combo} 评估失败: {str(e)}")

    # 保存结果
    with open('combination_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 显示最佳组合
    top_results = sorted(results, key=lambda x: x['mean_accuracy'], reverse=True)[:10]
    print("\n最佳特征组合 Top 10:")
    for i, res in enumerate(top_results, 1):
        print(f"{i:2}. {res['combination']} => Acc: {res['mean_accuracy']:.3f}±{res['std_accuracy']:.3f}")


if __name__ == '__main__':
    # 设置最大组合特征数（根据计算资源调整）
    evaluate_combinations(max_features=3)