import numpy as np


class KalmanFilter():

    def __init__(self, 
                 transition_matrix, observation_matrix,
                 transition_covariance, observation_covariance,
                 precompute: bool = True, index: int = 0):
        
        '''
        Args:
            transition matrix: Numpy array of shape (state_dim, state_dim)
            observation matrix: Numpy array of shape (meas_dim, state_dim)
            transition_covariance: Numpy array of shape (state_dim, state_dim)
            observation_covariance: Numpy array of shape (meas_dim, meas_dim)
            precompute:
            index:

        Returns: 
        '''
        
        self.F = transition_matrix
        self.H = observation_matrix
        self.Q = transition_covariance
        self.R = observation_covariance
        self.res = []
        self.tr_noise = []

        self.dim_measure, self.dim_state = observation_matrix.shape

        if precompute:
            self.I = np.eye(self.dim_state)
        else:
            raise NotImplementedError

        self._index = index


    def initialize_state(self, state, state_covariance=None):
        
        self.state = state
        self.state_covariance = state_covariance

    def filter_step(self, new_observation, return_predictions: bool = False):
        
        predicted_state, predicted_covariance = self.predict()
        updated_state, updated_covariance = self.update(new_observation)

        if not return_predictions:
            return updated_state, updated_covariance
        else:
            return updated_state, updated_covariance, predicted_state, predicted_covariance

    def predict(self):
        self.state = self.F @ self.state
        # Increase uncertainty
        self.state_covariance = self.F @ self.state_covariance @ self.F.T + self.Q

        return self.state, self.state_covariance

    def update(self, new_observation):
        # Predicted covariance of new measurement
        S = self.H @ self.state_covariance @ self.H.T + self.R 
        # Kalman gain (how much we should trust the new info)
        K = self.state_covariance @ self.H.T @ np.linalg.inv(S)
        # Innovation (additional information brought by new measurement)
        v = new_observation - self.H @ self.state        

        self.state = self.state + K @ v
        # Decrease uncertainty
        self.state_covariance = (self.I - K @ self.H) @ self.state_covariance


        return self.state, self.state_covariance


    @property
    def index(self):
        return self._index
