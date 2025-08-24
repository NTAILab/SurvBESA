import numpy as np
import torch
import cvxpy as cp
from beran import Beran
from pandas.core.common import flatten
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


import numpy as np
import sklearn
from pandas.core.common import flatten
import numpy as np
import pandas as pd
import scipy
import sksurv
import seaborn as sns
from tqdm import tqdm
import optuna

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

import matplotlib.pyplot as plt

from sksurv.datasets import load_whas500, load_aids, load_breast_cancer
from sksurv.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import brier_score, integrated_brier_score
from sksurv.datasets import load_gbsg2, load_veterans_lung_cancer
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from scipy.special import gamma as gamma_function

class EnsembleBeran():
    def __init__(self, omega = 1, tau = 1, maximum_number_of_pairs = 10,
                 n_estimators = 3, size_bagging = 0.4,
                 epsilon = 0.5, lr = 1e-1, const_in_div = 100, num_epoch = 100,
                 MAE_optimisation = True, 
                 epsilon_optimisation = True, 
                 c_index_optimisation = True,
                 use_attention = True,
                 use_knn = True,
                 mode = 'gradient') -> None:
        
        self.maximum_number_of_pairs = maximum_number_of_pairs
        self.n_estimators = n_estimators
        self.epsilon = epsilon
        self.size_bagging = size_bagging
        self.MAE_optimisation = MAE_optimisation
        self.epsilon_optimisation = epsilon_optimisation
        self.c_index_optimisation = c_index_optimisation
        self.omega = omega
        self.lr = lr
        self.const_in_div = const_in_div
        self.num_epoch = num_epoch
        self.mode = mode
        self.tau = tau
        self.use_attention = use_attention
        self.use_knn = use_knn
        np.random.seed(42)
        torch.manual_seed(42)
    
    def _sorting(self, X_train, y_train):
        times = np.array(list(zip(*y_train))[1])
        args = np.argsort(times)
        times = times[args]
        X_train = X_train[args]
        y_train = y_train[args]
        delta = np.array(list(zip(*y_train))[0])
        return X_train, y_train, times, delta
    
    def _select_random_points_with_neighbors(self, X_train, y_train, count_clusters, count_neighbors):
        valid_indices = np.where([status for status, _ in y_train])[0]  # выбираем только те индексы, у которых status = True
    
        # Если количество допустимых центров меньше чем count_clusters, возвращаем ошибку
        if len(valid_indices) < count_clusters:
            raise ValueError("Недостаточно элементов с status=True для выбора центров кластеров")

        # Случайным образом выбираем индексы центров из valid_indices
        random_indices = np.random.choice(valid_indices, count_clusters, replace=False)
        knn = NearestNeighbors(n_neighbors=count_neighbors+1)
        knn.fit(X_train)
        # print(random_indices)
        clusters = []

        for idx in random_indices:
            neighbors = knn.kneighbors([self.X_train[idx]], return_distance=False)
            neighbors[0] = np.sort(neighbors[0])
            neighbor_indices = neighbors[0]
            clusters.append(neighbor_indices)
        return clusters
    
    def _select_random_point(self, X_train, count_clusters, count_neighbors):
        clusters = []
        for _ in range(0, count_clusters, 1):
            # Случайно выбираем соседей для текущего кластера (включая сам центр)
            neighbors = np.random.choice(len(X_train), count_neighbors, replace=False)
            neighbors = np.sort(neighbors)  # Упорядочиваем для консистентности
            clusters.append(neighbors)

        return clusters
    
    def _find_set_for_C_index_optimisation(self, y_train, n):
        left = []
        right = []
        for i in range(0, y_train.shape[0]):
            if y_train[i][0]:
                current_right = []
                for j in range(0, y_train.shape[0]):
                    if y_train[j][1]>y_train[i][1]:
                        current_right.append(j)
                        left.append(i)
                if len(current_right) > n:
                    right.append(np.random.choice(current_right, size=n, replace=False))
                    left = left[:-(len(current_right)-n)]
                elif len(current_right) <= n and len(current_right)!=0:
                    right.append(current_right)
        right = list(flatten(right))
        left = np.array(left)
        right = np.array(right)
        return left, right
    
    def _train_each_beran(self, X_train, y_train, n_estimators, cluster_points, kernels, times_train, delta_train, const_in_div, C_index_optimisation, MAE_optimisation, tau):
        beran_models = []
        # print(const_in_div)
        for iteration in range(n_estimators):
            C_index_optimisation_k = C_index_optimisation
            MAE_optimisation_k = MAE_optimisation
            indeces = cluster_points[iteration]
            X_train_k = X_train[indeces]
            y_train_k = y_train[indeces]
            delta_k = delta_train[indeces]
            times_k = times_train[indeces]
            left_k = []
            right_k = []
            if C_index_optimisation == True:
                left_k, right_k = self._find_set_for_C_index_optimisation(y_train_k, self.maximum_number_of_pairs)
            # print(indeces)
            kernel_k = kernels[iteration]
            if len(left_k) == 0:
                C_index_optimisation_k = False
                MAE_optimisation_k = False
            beran_k = Beran(kernel_k, const_in_div, tau, C_index_optimisation_k, MAE_optimisation_k)
            beran_k.train(X_train_k, times_k, delta_k, left_k, right_k)
            beran_models.append(beran_k)
        return beran_models
    
    def _find_attention_S(self, S, omega):
        S_expanded_1 = S[:, :, np.newaxis, :]  # (n, k, 1, m)
        S_expanded_2 = S[:, np.newaxis, :, :]  # (n, 1, k, m)
    
        distances = np.max(np.abs(S_expanded_1 - S_expanded_2), axis=-1)  # (n, k, k)
        # print(S)
        n, k, _ = distances.shape
        diagonal_mask = np.eye(k, dtype=bool)[np.newaxis, :, :]  # (1, k, k), расширено до (n, k, k)

        # Устанавливаем -np.inf только для диагональных элементов
        distances = np.where(diagonal_mask, -np.inf, distances)
        # print(distances)
        weights = -(distances**2) / omega  # (n, k, k)

        #m*k*t
        # print(weights)
        weights_max = np.amax(weights, axis=2, keepdims=True)
        exp_weights_shifted = np.exp(weights-weights_max)
        # print(np.sum(exp_weights_shifted, axis=-1, keepdims=True))
        alpha = exp_weights_shifted / np.sum(exp_weights_shifted, axis=-1, keepdims=True)  # (n, k, k)
        # alpha[np.isnan(alpha)] = 1/S.shape[1]
        # print(alpha)
        smoothed_S = np.matmul(alpha, S)  # (n, k, m)
        return smoothed_S, alpha

    def _find_H_S_T(self, X_train, times_train, X_test, beran_models, n_estimators, cluster_points):
        H = []
        S = []
        T = []
        delta_times = np.diff(times_train)
        for iteration in range(n_estimators):
            beran = beran_models[iteration]
            H_k = np.zeros((X_test.shape[0], X_train.shape[0]))
            indeces = cluster_points[iteration]
            H_k[:, indeces] = beran.predict_cumulative_risk(X_test)
            S_k = np.exp(-np.cumsum(H_k, axis=1))
            S_k[:, -1] = 0
            # print(np.unique(S_k[1, :]))
            S_k_without_last = S_k[:, :-1]
            T_k = np.einsum('mt,t->m', S_k_without_last, delta_times)
            H.append(H_k)
            S.append(S_k)
            T.append(T_k)
        return np.stack(H, axis=1), np.stack(S, axis=1), np.stack(T, axis=1)
    
    def _find_attention_H_S_T(self, X, H, S, T, prototype, epsilon, v, gamma):
        alpha = self._ensemble_weights(prototype, X, gamma)

        attention_H = np.einsum('mkn,mk->mn', H, (1-epsilon)*alpha+epsilon*v)
        attention_S = np.einsum('mkn,mk->mn', S, (1-epsilon)*alpha+epsilon*v)
        attention_answer_T = np.einsum('mk,mk->m', T, (1-epsilon)*alpha+epsilon*v)

        return attention_H, attention_S, attention_answer_T

    def _find_attention_T(self, S_mean, times_train):
        delta_times = np.diff(times_train)
        S_mean_without_last = S_mean[:, :-1]
        T = np.einsum('mt,t->m', S_mean_without_last, delta_times)
        return T
    
    def _optimisation(self, alpha, S, times_train, left, right,
                      epsilon, lr, const_in_div, num_epoch):#изменить
        
        epsilon = torch.tensor(epsilon, device="cpu")  
        R = torch.rand((alpha.shape[1], alpha.shape[2]), requires_grad=True, device="cpu", dtype=torch.float64)
        optimizer = torch.optim.Adam([R], lr=lr)
        S_tensor = torch.tensor(S, device="cpu")
        times_tensor = torch.tensor(times_train, device="cpu")
        for iteration in range(num_epoch): #изменить
            optimizer.zero_grad()

            alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device="cpu")
            # R_modified = R.clone()
            # R_modified[torch.eye(R.shape[0], dtype=bool)] = -np.inf
            sm = torch.nn.Softmax(dim=1)
            R_softmax = sm(R) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # print(R)
            R_tensor = R_softmax.unsqueeze(0)
            weights = epsilon*alpha_tensor + (1-epsilon)*R_tensor
            smoothed_S = torch.matmul(weights, S_tensor)
            S_mean = torch.mean(smoothed_S, dim=1) #n*t
            delta_times = torch.diff(times_tensor)
            S_mean_without_last = S_mean[:, :-1]
            T_tensor = torch.einsum('mt,t->m', S_mean_without_last, delta_times)
            # print(left)
            # print(right)
            G_ij = -T_tensor[left] + T_tensor[right]
            G_ij = G_ij/const_in_div
            loss = torch.sum(1 / (1 + torch.exp(G_ij)))/left.shape[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_([R], 100)

            optimizer.step()
            with torch.no_grad():
                torch.nan_to_num_(R.data, nan=0.0)
        return R_softmax.detach().cpu().numpy(), epsilon.item()
    
    def fit(self, X_train, y_train):#изменить
        times = np.array(list(zip(*y_train))[1])
        args = np.unique(times, return_index=True)[1]
        times = times[args]
        y_train = y_train[args]
        X_train = X_train[args]
        self.X_train, self.y_train, self.times_train, self.delta_train = self._sorting(X_train, y_train)
        self.left, self.right = self._find_set_for_C_index_optimisation(self.y_train, self.maximum_number_of_pairs)

        list_of_kernels = ["gauss"]
        number_of_items_in_subsample = int(self.X_train.shape[0]*self.size_bagging)
        if self.use_knn == True:
            self.cluster_points = self._select_random_points_with_neighbors(self.X_train, self.y_train, self.n_estimators, number_of_items_in_subsample)
        else:
            self.cluster_points = self._select_random_point(self.X_train, self.n_estimators, number_of_items_in_subsample)
        kernels = np.random.choice(list_of_kernels, self.n_estimators)
        self.beran_models = self._train_each_beran(self.X_train, self.y_train, self.n_estimators, self.cluster_points, kernels,
                                                      self.times_train, self.delta_train,
                                                      self.const_in_div, self.c_index_optimisation, self.MAE_optimisation, self.tau)
        H, S, T = self._find_H_S_T(self.X_train, self.times_train, self.X_train, self.beran_models, 
                                   self.n_estimators, self.cluster_points)
        
        if self.use_attention:
            S_att, alpha = self._find_attention_S(S, omega = self.omega)
            if self.c_index_optimisation:
                self.R, epsilon = self._optimisation(alpha, S, self.times_train, self.left, self.right, self.epsilon, self.lr, self.const_in_div, self.num_epoch)#изменить
                weights = epsilon*alpha + (1-epsilon)*self.R
                S_opt = np.matmul(weights, S)
                S_mean_opt = np.mean(S_opt, axis=1)

    def _predict(self, X_test, R):#изменить
        H, S, T = self._find_H_S_T(self.X_train, self.times_train, X_test, self.beran_models, 
                                   self.n_estimators, self.cluster_points)
        if self.use_attention:
            S_smoothed, alpha = self._find_attention_S(S, omega = self.omega)
            weights = self.epsilon*alpha + (1-self.epsilon)*R
            smoothed_S = np.matmul(weights, S)
            S_mean = np.mean(smoothed_S, axis=1)
            T_att = self._find_attention_T(S_mean, self.times_train)
            return T_att
    
    def predict(self, X_test):
        H, S, T = self._find_H_S_T(self.X_train, self.times_train, X_test, self.beran_models, 
                                   self.n_estimators, self.cluster_points)
        if self.use_attention:
            S_smoothed, alpha = self._find_attention_S(S, omega = self.omega)
            if self.c_index_optimisation:
                weights = self.epsilon*alpha + (1-self.epsilon)*self.R
                smoothed_S = np.matmul(weights, S)
                S_mean = np.mean(smoothed_S, axis=1)
                T_att_opt = self._find_attention_T(S_mean, self.times_train)
                return T_att_opt
            else:
                S_mean = np.mean(S_smoothed, axis=1)
                T_att = self._find_attention_T(S_mean, self.times_train)
                return T_att
        else:
            T_mean = np.mean(T, axis=1)
            return T_mean
