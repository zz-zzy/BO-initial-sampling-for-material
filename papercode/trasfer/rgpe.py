
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import matplotlib.pyplot as plt


from utils import sample_sobol

from collections import Counter

import GPy

from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper


def plot_predictions_and_train_y(predictions, train_y):

    # 计算每一行数据的最小值和最大值
    min_values = np.min(predictions, axis=1)
    max_values = np.max(predictions, axis=1)

    # 对每一行数据进行 Min-Max 归一化处理
    normalized_predictions = (predictions - min_values[:, np.newaxis]) / (max_values - min_values)[:, np.newaxis]
    normalized_train_y = (train_y - np.min(train_y)) / (np.max(train_y) - np.min(train_y))

    print(normalized_predictions)

    # 配置字体为Arial
    plt.rcParams["font.family"] = "Arial"
    # 创建一个颜色循环，以便为每条折线选择不同的颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(normalized_predictions) - 1))  # 除了最后一行的颜色

    # 创建一个图形
    plt.figure(figsize=(10, 6))

    # # 遍历前面的行数据并绘制折线
    # for i, row in enumerate(normalized_predictions[:-1]):
    #     plt.plot(row, color=colors[i], marker='o', markersize=6, label=f'Source {i + 1}')

    # 绘制最后一行（predictions 中的最后一行）并用红色表示
    plt.plot(normalized_predictions[-1], color='red', marker='o', markersize=6, label='Target')

    # 添加 train_y 到折线图中并用黑色表示
    plt.plot(normalized_train_y, color='black', marker='o', markersize=6, label='Train Y')

    # 添加图例
    plt.legend()

    # 添加坐标轴标签
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')

    # 显示图形
    plt.grid()
    plt.show()





def roll_col(X: np.ndarray, shift: int) -> np.ndarray:
    """
    Rotate columns to right by shift.
    """
    return np.concatenate((X[:, -shift:], X[:, :-shift]), axis=1)






def compute_ranking_loss(
    f_samps: np.ndarray,
    target_y: np.ndarray,
    target_model: bool,
) -> np.ndarray:
    """
    Compute ranking loss for each sample from the posterior over target points.
    """
    y_stack = np.tile(target_y.reshape((-1, 1)), f_samps.shape[0]).transpose()
    rank_loss = np.zeros(f_samps.shape[0])
    if not target_model:
        for i in range(1, target_y.shape[0]):
            rank_loss += np.sum(
                (roll_col(f_samps, i) < f_samps) ^ (roll_col(y_stack, i) < y_stack),  #用矩阵滚动的形式，让每个数都进行了比较
                axis=1
            )
    else:
        for i in range(1, target_y.shape[0]):
            rank_loss += np.sum(
                (roll_col(f_samps, i) < y_stack) ^ (roll_col(y_stack, i) < y_stack),
                axis=1
            )

    return rank_loss


def sample_random(model, X, num_samples, seed):
    np.random.seed(seed)  # 设置随机数种子

    samples = np.empty((len(X), num_samples))  # 创建空的抽样矩阵

    for i in range(len(X)):
        mu, var = model.predict(np.array([X[i]]))  # 使用模型预测 X[i] 处的 mu 和 var

        # 根据 mu 和 var 进行抽样，并将结果赋值给抽样矩阵的相应位置
        samples[i, :] = np.random.normal(mu[0, 0],var[0, 0], num_samples) #[:, np.newaxis]

    return samples





def get_target_model_loocv_sample_preds(
    train_x: np.ndarray,
    train_y: np.ndarray,
    num_samples: int,
    target_model,
    engine_seed: int,
) -> np.ndarray:
    """
    Use LOOCV to fit len(train_y) independent GPs and sample from their posterior to obtain an
    approximate sample from the target model.

    This sampling does not take into account the correlation between observations which occurs
    when the predictive uncertainty of the Gaussian process is unequal zero.
    """
    masks = np.eye(len(train_x), dtype=np.bool)
    train_x_cv = np.stack([train_x[~m] for m in masks])
    train_y_cv = np.stack([train_y[~m] for m in masks])
    test_x_cv = np.stack([train_x[m] for m in masks])

    samples = np.zeros((num_samples, train_y.shape[0]))
    for i in range(train_y.shape[0]):

        loo_model = target_model
        loo_model.set_data(train_x_cv[i], train_y_cv[i])




        samples_i = sample_sobol(loo_model, test_x_cv[i], num_samples, engine_seed).flatten()

        samples[:, i] = samples_i

    return samples


def compute_target_model_ranking_loss(
    train_x: np.ndarray,
    train_y: np.ndarray,
    num_samples: int,
    target_model,
    engine_seed: int,
) -> np.ndarray:
    """
    Use LOOCV to fit len(train_y) independent GPs and sample from their posterior to obtain an
    approximate sample from the target model.

    This function does joint draws from all observations (both training data and left out sample)
    to take correlation between observations into account, which can occur if the predictive
    variance of the Gaussian process is unequal zero. To avoid returning a tensor, this function
    directly computes the ranking loss.
    """
    masks = np.eye(len(train_x), dtype=np.bool)
    train_x_cv = np.stack([train_x[~m] for m in masks])
    train_y_cv = np.stack([train_y[~m] for m in masks])

    ranking_losses = np.zeros(num_samples, dtype=np.int)
    for i in range(train_y.shape[0]):

        loo_model = target_model
        loo_model.set_data(train_x_cv[i], train_y_cv[i])



        samples_i = sample_sobol(loo_model, train_x, num_samples, engine_seed)

        for j in range(len(train_y)):
            ranking_losses += (samples_i[:, i] < samples_i[:, j]) ^ (train_y[i] < train_y[j])

    return ranking_losses


def compute_rank_weights(
        train_x: np.ndarray,
        train_y: np.ndarray,
        base_models,
        target_model,
        num_samples: int,  # 抽取次数，每个点都抽一个 代表一次模型抽样
        sampling_mode: str,
        weight_dilution_strategy: Union[int, Callable],
        number_of_function_evaluations,
        rng: np.random.RandomState,
        alpha: float = 0.0,
        plot_target_pred_vs_true=False
) -> np.ndarray:
    """
    Compute ranking weights for each base model and the target model
    (using LOOCV for the target model).

    Returns
    -------
    weights : np.ndarray
    """

    if sampling_mode == 'bootstrap': #第一种抽样策略

        predictions = []
        for model_idx in range(len(base_models)):
            model = base_models[model_idx]
            predictions.append(model.predict(train_x)[0].flatten())   #每个base模型都给出预测

        masks = np.eye(len(train_x), dtype=np.bool)
        train_x_cv = np.stack([train_x[~m] for m in masks])
        train_y_cv = np.stack([train_y[~m] for m in masks])
        test_x_cv = np.stack([train_x[m] for m in masks]) # target模型 其他点做训练，其中一个点做预测

        loo_prediction = []
        for i in range(train_y.shape[0]):
            loo_model = target_model
            loo_model.set_data(train_x_cv[i], train_y_cv[i])

            #
            # f_obj = loo_model.model.predict
            # loo_prediction.append(f_obj(test_x_cv[i])[0][0][0])


#检验loo_model.predict(test_x_cv[i])[0][0][0]的作用
            # a = loo_model.predict(test_x_cv[i])
            # b = loo_model.predict(test_x_cv[i])[0]
            # bb = loo_model.predict(test_x_cv[i])[0].flatten()
            # c = loo_model.predict(test_x_cv[i])[0][0]
            # d = loo_model.predict(test_x_cv[i])[0][0][0]

            loo_prediction.append(loo_model.predict(test_x_cv[i])[0][0][0])  #每个loo target模型给出预测   为啥写这么麻烦我也不懂
        predictions.append(loo_prediction)
        predictions = np.array(predictions) # base 和 target 的总预测，预测出来是应该是 mu

        # # 绘制源模型和留一模型对观察点预测值的minmax归一化折线图
        if plot_target_pred_vs_true:
            plot_predictions_and_train_y(predictions, train_y)

        bootstrap_indices = rng.choice(predictions.shape[1],
                                       size=(num_samples, predictions.shape[1]),
                                       replace=True)  # 对predictions.shape[1]的维度个数进行抽样，比如predictions.shape[1] = 10 ，则从0-9抽数，所以出来是索引

        bootstrap_predictions = []
        bootstrap_targets = train_y[bootstrap_indices].reshape((num_samples, len(train_y))) # 抽样对应的y值
        for m in range(len(base_models) + 1):
            bootstrap_predictions.append(predictions[m, bootstrap_indices])

        ranking_losses = np.zeros((len(base_models) + 1, num_samples))
        for i in range(len(base_models)):

            for j in range(len(train_y)):
#检验ranking_losses的计算方法
                # a = bootstrap_predictions[i]
                # b = roll_col(bootstrap_predictions[i], j)
                # c = (roll_col(bootstrap_predictions[i], j) < bootstrap_predictions[i]) ^ (roll_col(bootstrap_targets, j) < bootstrap_targets)
                # c1 = (roll_col(bootstrap_predictions[i], j) < bootstrap_predictions[i])
                # c2 = (roll_col(bootstrap_targets, j) < bootstrap_targets)
                # d = np.sum(
                #     (
                #         roll_col(bootstrap_predictions[i], j) < bootstrap_predictions[i])
                #         ^ (roll_col(bootstrap_targets, j) < bootstrap_targets
                #     ), axis=1)


                ranking_losses[i] += np.sum(
                    (
                        roll_col(bootstrap_predictions[i], j) < bootstrap_predictions[i])
                        ^ (roll_col(bootstrap_targets, j) < bootstrap_targets
                    ), axis=1
                )
        for j in range(len(train_y)):
            ranking_losses[-1] += np.sum(
                (
                    (roll_col(bootstrap_predictions[-1], j) < bootstrap_targets)   #这里为什么不是 bootstrap_predictions[-1] ？
                    ^ (roll_col(bootstrap_targets, j) < bootstrap_targets)
                ), axis=1
            )

    elif sampling_mode in ['simplified', 'correct']: #第二种抽样策略
        # Use the original strategy as described in v1: https://arxiv.org/pdf/1802.02219v1.pdf
        ranking_losses = []
        # compute ranking loss for each base model
        for model_idx in range(len(base_models)):
            model = base_models[model_idx]
            # compute posterior over training points for target task
            f_samps = sample_sobol(model, train_x, num_samples, rng.randint(10000))
            # compute and save ranking loss
            ranking_losses.append(compute_ranking_loss(f_samps, train_y, target_model=False))

        # compute ranking loss for target model using LOOCV
        if sampling_mode == 'simplified':
            # Independent draw of the leave one out sample, other "samples" are noise-free and the
            # actual observation
            f_samps = get_target_model_loocv_sample_preds(train_x, train_y, num_samples,target_model,
                                                          rng.randint(10000))
            ranking_losses.append(compute_ranking_loss(f_samps, train_y, target_model=True))
        elif sampling_mode == 'correct':
            # Joint draw of the leave one out sample and the other observations
            ranking_losses.append(
                compute_target_model_ranking_loss(train_x, train_y, num_samples,target_model,
                                                  rng.randint(10000))
            )
        else:
            raise ValueError(sampling_mode)
    else:
        raise NotImplementedError(sampling_mode)
# ranking loss 计算结束

    if isinstance(weight_dilution_strategy, int):
        weight_dilution_percentile_target = weight_dilution_strategy
        weight_dilution_percentile_base = 50
    elif weight_dilution_strategy is None or weight_dilution_strategy in ['probabilistic', 'probabilistic-ld']:
        pass
    else:
        raise ValueError(weight_dilution_strategy)

    ranking_loss = np.array(ranking_losses)

    # perform model pruning
    p_drop = []
    if weight_dilution_strategy in ['probabilistic', 'probabilistic-ld']:
        for i in range(len(base_models)):
            better_than_target = np.sum(ranking_loss[i, :] < ranking_loss[-1, :])
            worse_than_target = np.sum(ranking_loss[i, :] >= ranking_loss[-1, :])
            correction_term = alpha * (better_than_target + worse_than_target)
            proba_keep = better_than_target / (better_than_target + worse_than_target + correction_term)
            if weight_dilution_strategy == 'probabilistic-ld':
                proba_keep = proba_keep * (1 - len(train_x) / float(number_of_function_evaluations))
            proba_drop = 1 - proba_keep
            p_drop.append(proba_drop)
            r = rng.rand()
            if r < proba_drop:
                ranking_loss[i, :] = np.max(ranking_loss) * 2 + 1
    elif weight_dilution_strategy is not None:
        # Use the original strategy as described in v1: https://arxiv.org/pdf/1802.02219v1.pdf
        percentile_base = np.percentile(ranking_loss[: -1, :], weight_dilution_percentile_base, axis=1)
        percentile_target = np.percentile(ranking_loss[-1, :], weight_dilution_percentile_target)
        for i in range(len(base_models)):
            if percentile_base[i] >= percentile_target:
                ranking_loss[i, :] = np.max(ranking_loss) * 2 + 1
#防止权重稀释策略结束
    # compute best model (minimum ranking loss) for each sample
    # this differs from v1, where the weight is given only to the target model in case of a tie.
    # Here, we distribute the weight fairly among all participants of the tie.
    minima = np.min(ranking_loss, axis=0)
    assert len(minima) == num_samples
    best_models = np.zeros(len(base_models) + 1)
    for i, minimum in enumerate(minima):
        minimum_locations = ranking_loss[:, i] == minimum  #找到最小 则为true
        sample_from = np.where(minimum_locations)[0]

        for sample in sample_from:
            best_models[sample] += 1. / len(sample_from)  #计数true的数量

    # compute proportion of samples for which each model is best
    rank_weights = best_models / num_samples
    return rank_weights  #, p_drop


# from emukit.test_functions import forrester_function
# from pyDOE import lhs
# target_function, space = forrester_function()
#
#
# X  = lhs(len(space.parameters), 3, criterion='maximin')
# # X = np.array(np.random.rand(2, 1))
# # X = np.append(X, x_new_1, axis=0)
# Y = target_function(X)
#
# base_models = [base_model]
# rng = np.random.RandomState(1)
# n_samples = 1000
# sampling_mode = 'correct'
#
# weight_list = compute_rank_weights(X, Y, n_samples,sampling_mode,rng)
# w_1 = weight_list[0]
# w_t = weight_list[-1]
#
# print(weight_list)
# print(w_1)
# print(w_t)

'''
class RGPE(AbstractEPM):

    def __init__(
        self,
        training_data: Dict[int, Dict[str, Union[List[Configuration], np.ndarray]]],
        num_posterior_samples: int,
        weight_dilution_strategy: Union[int, str],
        number_of_function_evaluations: int,
        sampling_mode: str = 'correct',
        variance_mode: str = 'average',
        normalization: str = 'mean/var',
        alpha: float = 0.0,
        **kwargs
    ):
        """Ranking-Weighted Gaussian Process Ensemble.

        Parameters
        ----------
        training_data
            Dictionary containing the training data for each meta-task. Mapping from an integer (
            task ID) to a dictionary, which is a mapping from configuration to performance.
        num_posterior_samples
            Number of samples to draw for approximating the posterior probability of a model
            being the best model to explain the observations on the target task.
        weight_dilution_strategy
            Can be one of the following four:
            * ``'probabilistic-ld'``: the method presented in the paper
            * ``'probabilistic'``: the method presented in the paper, but without the time-dependent
              pruning of meta-models
            * an integer: a deterministic strategy described in https://arxiv.org/abs/1802.02219v1
            * ``None``: no weight dilution prevention
        number_of_function_evaluations
            Optimization horizon - used to compute the time-dependent factor in the probability
            of dropping base models for the weight dilution prevention strategy
            ``'probabilistic-ld'``.
        sampling_mode
            Can be any of:
            * ``'bootstrap'``
            * ``'correct'``
            * ``'simplified'``
        variance_mode
            Can be either ``'average'`` to return the weighted average of the variance
            predictions of the individual models or ``'target'`` to only obtain the variance
            prediction of the target model. Changing this is only necessary to use the model
            together with the expected improvement.
        normalization
            Can be either:
            * ``None``: No normalization per task
            * ``'mean/var'``: Zero mean unit standard deviation normalization per task as
              proposed by Yogatama et al. (AISTATS 2014).
            * ``'Copula'``: Copula transform as proposed by Salinas et al., 2020
        alpha
            Regularization hyperparameter to increase aggressiveness of dropping base models when
            using the weight dilution strategies ``'probabilistic-ld'`` or ``'probabilistic'``.
        """

        if kwargs.get('instance_features') is not None:
            raise NotImplementedError()
        super().__init__(**kwargs)
        self.training_data = training_data

        self.number_of_function_evaluations = number_of_function_evaluations
        self.num_posterior_samples = num_posterior_samples
        self.rng = np.random.RandomState(self.seed)
        self.sampling_mode = sampling_mode
        self.variance_mode = variance_mode
        self.normalization = normalization
        self.alpha = alpha

        if self.normalization not in ['None', 'mean/var', 'Copula']:
            raise ValueError(self.normalization)

        if weight_dilution_strategy is None or weight_dilution_strategy == 'None':
            weight_dilution_strategy = None
        elif weight_dilution_strategy in ['probabilistic', 'probabilistic-ld']:
            pass
        else:
            weight_dilution_strategy = int(weight_dilution_strategy)

        self.weight_dilution_strategy = weight_dilution_strategy

        base_models = []  # 对base的y进行归一化，训练并生成基础模型
        for task in training_data:
            model = get_gaussian_process(
                bounds=self.bounds,
                types=self.types,
                configspace=self.configspace,
                rng=self.rng,
                kernel=None,
            )
            y = training_data[task]['y']
            if self.normalization == 'mean/var':
                mean = y.mean()
                std = y.std()
                if std == 0:
                    std = 1

                y_scaled = (y - mean) / std
                y_scaled = y_scaled.flatten()
            elif self.normalization == 'Copula':
                y_scaled = copula_transform(y)
            elif self.normalization == 'None':
                y_scaled = y
            else:
                raise ValueError(self.normalization)
            configs = training_data[task]['configurations']
            X = convert_configurations_to_array(configs)

            model.train(
                X=X,
                Y=y_scaled,
            )
            base_models.append(model)
        self.base_models = base_models
        self.weights_over_time = []
        self.p_drop_over_time = []

    def _train(self, X: np.ndarray, Y: np.ndarray) -> AbstractEPM: #对target的y归一化，训练target model，并计算权重和drop
        """SMAC training function"""
        print(self.normalization)
        if self.normalization == 'mean/var':
            Y = Y.flatten()
            mean = Y.mean()
            std = Y.std()
            if std == 0:
                std = 1

            y_scaled = (Y - mean) / std
            self.Y_std_ = std
            self.Y_mean_ = mean
        elif self.normalization in ['None', 'Copula']:
            self.Y_mean_ = 0.
            self.Y_std_ = 1.
            y_scaled = Y
            if self.normalization == 'Copula':
                y_scaled = copula_transform(Y)
        else:
            raise ValueError(self.normalization)  #像base model一样，对y进行处理

        target_model = get_gaussian_process(
            bounds=self.bounds,
            types=self.types,
            configspace=self.configspace,
            rng=self.rng,
            kernel=None,
        )
        self.target_model = target_model.train(X, y_scaled) #训练target model
        self.model_list_ = self.base_models + [target_model]

        if X.shape[0] < 3:
            self.weights_ = np.ones(len(self.model_list_)) / len(self.model_list_) #x个数小于3，平均分配权重
            p_drop = np.ones((len(self.base_models, ))) * np.NaN
        else:
            try:
                self.weights_, p_drop = compute_rank_weights(
                    train_x=X,
                    train_y=y_scaled,
                    base_models=self.base_models,
                    target_model=target_model,
                    num_samples=self.num_posterior_samples,
                    sampling_mode=self.sampling_mode,
                    weight_dilution_strategy=self.weight_dilution_strategy,
                    number_of_function_evaluations=self.number_of_function_evaluations,
                    rng=self.rng,
                    alpha=self.alpha,
                )
            except Exception as e:
                print(e)
                self.weights_ = np.zeros((len(self.model_list_, )))
                self.weights_[-1] = 1
                p_drop = np.ones((len(self.base_models, ))) * np.NaN

        print('Weights', self.weights_)
        self.weights_over_time.append(self.weights_)
        self.p_drop_over_time.append(p_drop)

        return self

    def _predict(self, X: np.ndarray, cov_return_type='diagonal_cov') -> Tuple[np.ndarray, np.ndarray]:#给出所有 model 预测的x处的加权 mu和var
        """SMAC predict function"""

        # compute posterior for each model
        weighted_means = []
        weighted_covars = []

        # filter model with zero weights
        # weights on covariance matrices are weight**2
        non_zero_weight_indices = (self.weights_ ** 2 > 0).nonzero()[0]
        non_zero_weights = self.weights_[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()

        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            weight = non_zero_weights[non_zero_weight_idx]
            mean, covar = self.model_list_[raw_idx]._predict(X, cov_return_type)

            weighted_means.append(weight * mean)

            if self.variance_mode == 'average':
                weighted_covars.append(covar * weight ** 2)
            elif self.variance_mode == 'target':
                if raw_idx + 1 == len(self.weights_):
                    weighted_covars.append(covar)
            else:
                raise ValueError()

        if len(weighted_covars) == 0: #检查base model
            if self.variance_mode != 'target':
                raise ValueError(self.variance_mode)
            _, covar = self.model_list_[-1]._predict(X, cov_return_type=cov_return_type)
            weighted_covars.append(covar)

        mean_x = np.sum(np.stack(weighted_means), axis=0) * self.Y_std_ + self.Y_mean_
        covar_x = np.sum(weighted_covars, axis=0) * (self.Y_std_ ** 2)
        return mean_x, covar_x

    def sample_functions(self, X_test: np.ndarray, n_funcs: int = 1) -> np.ndarray: #对非0的base model采样X_test的函数值，加权求和
        """
        Sample function values from the posterior of the specified test points.
        """

        # filter model with zero weights
        # weights on covariance matrices are weight**2
        non_zero_weight_indices = (self.weights_ ** 2 > 0).nonzero()[0]
        non_zero_weights = self.weights_[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()

        samples = []
        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            weight = non_zero_weights[non_zero_weight_idx]

            funcs = self.model_list_[raw_idx].sample_functions(X_test, n_funcs)
            funcs = funcs * weight
            samples.append(funcs)
        samples = np.sum(samples, axis=0)
        return samples
'''