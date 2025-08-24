import numpy as np

class Beran():
    def __init__(self, kernel, const_in_div, tau, C_index_optimisation, MAE_optimisation) -> None:
        self.kernel = kernel
        self.const_in_div = const_in_div
        self.C_index_optimisation = C_index_optimisation
        self.MAE_optimisation = MAE_optimisation
        self.tau = tau
        
    def _epanechnikov_kernel_normalized(self, weights):
        kernel = 0.75 * np.maximum(0, (1 - weights**2))
        kernel = np.nan_to_num(kernel, posinf=0.0)
        return kernel / np.sum(kernel, axis=1, keepdims=True)
    
    def _triangular_kernel_normalized(self, weights):
        kernel = np.maximum(0, 1 - np.abs(weights))
        kernel = np.nan_to_num(kernel, posinf=0.0)
        return kernel / np.sum(kernel, axis=1, keepdims=True)
    
    def _quartic_kernel_normalized(self, weights):
        kernel = (15 / 16) * (1 - weights**2)**2
        kernel = np.nan_to_num(kernel, posinf=0.0)
        return kernel / np.sum(kernel, axis=1, keepdims=True)
    
    def _gauss_kernel_normalized(self, weights):
        kernel = np.exp(weights)
        return kernel / np.sum(kernel, axis=1, keepdims=True)
    
    def _relative_weights(self, X_train, X_test, temperature):
        #n*m
        X_1 = X_train[np.newaxis, :, :]   #1*n*f
        X_2 = X_test[:, np.newaxis, :]   #m*1*f
        diff = X_2 - X_1    #m*n*f
        distance = np.linalg.norm(diff, axis=2) #m*n
        distance = np.where(distance != 0, distance, -np.inf)
        weights = -(distance**2)/temperature
        weights_max = np.amax(weights, axis=1, keepdims=True)
        weights_shifted = (weights - weights_max)
        if self.kernel == "epanechnikov":
            return self._epanechnikov_kernel_normalized(weights_shifted)
        
        if self.kernel == "triangular":
            return self._triangular_kernel_normalized(weights_shifted)
        
        if self.kernel == "quartic":
            return self._quartic_kernel_normalized(weights_shifted)
        
        if self.kernel == "gauss":
            return self._gauss_kernel_normalized(weights_shifted)
    
    def _find_cumulative_risk(self, alpha, delta):
        cumsum = np.cumsum(alpha, axis=1) # m
        s_cumsum = cumsum - alpha   # 
        H = delta[np.newaxis] * (-np.log(np.maximum(1-cumsum, 10e-20)) + np.log(np.maximum(1-s_cumsum, 10e-20)))
        return H
    
    def _find_survival_function(self, H):
        S = np.exp(-np.cumsum(H, axis=1))
        return S
    
    def _find_expected_time(self, S, times):
        S_without_last = S[:, :-1]
        delta_times = np.diff(times)
        T = np.einsum('mt,t->m', S_without_last, delta_times)
        return T
    
    def _loss_function_C_index(self, T, left, right, const_in_div):
        G_ij = T[left] - T[right]  # power J
        G_ij = G_ij/const_in_div
        return np.sum(1 / (1 + np.exp((G_ij))))/self.left.shape[0]
    
    def _loss_function_MAE(self, T, times_train, delta_train):
        return np.sum((T[delta_train] - times_train[delta_train])**2)
    
    def _optimisation_temperature(self, X_train, times_train, delta_train, 
                                  left, right, const_in_div, C_index_optimisation, MAE_optimisation):
        best_temperature = np.random.randint(1, 3)
        best_loss = float('-inf')
        temperature_values = np.array([10**(-1), 10**(0), 10**(1), 10**(2)])
        for temperature in temperature_values:
            alpha = np.array(self._relative_weights(X_train, X_train, temperature)) # m*n
            H = self._find_cumulative_risk(alpha, delta_train)
            S = self._find_survival_function(H)
            T = self._find_expected_time(S, times_train)
            if C_index_optimisation == True:
                loss = self._loss_function_C_index(T, left, right, const_in_div)
            elif MAE_optimisation == True:
                loss = -self._loss_function_MAE(T, times_train, delta_train)
            if loss > best_loss:
                best_loss = loss
                best_temperature = temperature
        return best_temperature

    def train(self, X_train, times_train, delta_train, left, right):
        self.X_train = X_train
        self.delta_train = delta_train
        self.times_train = times_train
        self.left = left
        self.right = right
        if self.C_index_optimisation == True or self.MAE_optimisation == True:
            self.tau = self._optimisation_temperature(X_train, times_train, delta_train, left, right,
                                                    self.const_in_div, self.C_index_optimisation, self.MAE_optimisation)
            # print(self.tau)
        return

    def predict_expected_time(self, X_test):
        alpha = np.array(self._relative_weights(self.X_train, X_test, self.tau))
        H = self._find_cumulative_risk(alpha, self.delta_train)
        S = self._find_survival_function(H)
        T = self._find_expected_time(S, self.times_train)
        return T
    
    def predict_survival_function(self, X_test):
        alpha = np.array(self._relative_weights(self.X_train, X_test, self.tau))
        H = self._find_cumulative_risk(alpha, self.delta_train)
        S = self._find_survival_function(H)
        return S
    
    def predict_cumulative_risk(self, X_test):
        alpha = np.array(self._relative_weights(self.X_train, X_test, self.tau))
        H = self._find_cumulative_risk(alpha, self.delta_train)
        return H
