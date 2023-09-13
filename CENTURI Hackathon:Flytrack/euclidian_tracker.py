import numpy as np
from tqdm import tqdm
from numpy.random import multivariate_normal

from tracking.kalman_filter import KalmanFilter

from scipy.spatial import KDTree as scipy_KDTree
from scipy.spatial import distance_matrix as scipy_distance_matrix
from scipy.optimize import linear_sum_assignment as lsa




class EuclidianTracker:

    def __init__(self, trackability=True, n_sampling=20):
        
        self.trackability = trackability
        self.n_sampling = n_sampling
        self.appearances = 0
        self.disappearances = 0

        if n_sampling == 0:
            self.trackability = False
        else:
            self.trackability = True


    def track(self, observations, observation_noise, transition_noise,
              states_initial_covariance, list_dts):

        """
        Parameters
        ----------
        observations : [n_timesteps, n_objects_at_t, n_dim_obs]
            observations for time steps [0...n_timesteps-1]

        observations_covariance_matrice: [n_timesteps, n_objects_at_t, n_dim_obs, n_dim_obs]

        states_initial_covariances: [n_objects_at_t0, n_dim_state, n_dim_state]

        list_dts: [n_timesteps - 1]
        """

        base_transition_matrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        observation_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        observations_covariance_matrice = np.array([
            [observation_noise**2, 0],
            [0, observation_noise**2]])


        tracks = []
        states = []
        trackability_scores = []

        index_track = 0

        transition_matrix = base_transition_matrix.copy()
        dt = list_dts[0]
        transition_matrix[:2, 2:] = transition_matrix[:2, 2:] * dt

        for initial_observation in observations[0]:
            
            transition_covariance_matrix = np.array([
                [(dt**4)/4, 0, (dt**3)/2, 0],
                [0, (dt**4)/4, 0, (dt**3)/2],
                [(dt**3)/2, 0, dt**2, 0],
                [0, (dt**3)/2, 0, dt**2]
                ]) * transition_noise**2


            kalman_filter = KalmanFilter(
                transition_matrix=transition_matrix,
                observation_matrix=observation_matrix,
                transition_covariance=transition_covariance_matrix,
                observation_covariance=observations_covariance_matrice,
                index=index_track
            )

            kalman_filter.initialize_state(
                state=np.array([*initial_observation, 0, 0]),
                state_covariance=states_initial_covariance
            )

            states.append(kalman_filter)

            tracks.append([[0, *initial_observation]])

            index_track += 1


        for ind_t, dt in enumerate(tqdm(list_dts), start=1):

            transition_covariance_matrix = np.array([
                [(dt**4)/4, 0, (dt**3)/2, 0],
                [0, (dt**4)/4, 0, (dt**3)/2],
                [(dt**3)/2, 0, dt**2, 0],
                [0, (dt**3)/2, 0, dt**2]
                ]) * transition_noise**2

            
            predictions_at_t = [tuple(state.predict()) for state in states]
            observations_at_t = observations[ind_t]

            predictions=[mean[:2] for mean, _ in predictions_at_t]

            inds_predictions_at_t = range(len(predictions_at_t))
            inds_states_at_t = [kf.index for kf in states]

            inds_observations_at_t = range(len(observations_at_t))

            distance_matrix = scipy_distance_matrix(np.stack(predictions), np.stack(observations_at_t))

            ### GATING USING CUTOFFS BASED ON PREDICTED COVARIANCE
            for ind_prediction, prediction in enumerate(predictions_at_t):

                line = distance_matrix[ind_prediction]

                pred_mean, pred_covariance = prediction

                cutoff = 3 * np.sqrt(
                    (pred_covariance[0,0] + pred_covariance[1,1])/2
                )

                line[line>cutoff] = 1e8

                distance_matrix[ind_prediction] = line
            ###



            ### SOLVE LSA
            inds_pred_pre, inds_obs_pre = lsa(distance_matrix**2)

            inds_pred = []
            inds_obs = []

            for i in range(len(inds_pred_pre)):
                if distance_matrix[inds_pred_pre[i], inds_obs_pre[i]] != 1e8:
                    inds_pred.append(inds_pred_pre[i])
                    inds_obs.append(inds_obs_pre[i])
            ###


            ######### trackability ##############
            if self.trackability:

                mc_preds = []

                for i in range(self.n_sampling):

                    predictions = [multivariate_normal(mean, cov)[:2] for mean, cov in predictions_at_t]
                    distance_matrix = scipy_distance_matrix(np.stack(predictions), np.stack(observations_at_t))

                    ### GATING USING CUTOFFS BASED ON PREDICTED COVARIANCE
                    for ind_prediction, prediction in enumerate(predictions_at_t):

                        line = distance_matrix[ind_prediction]

                        pred_mean, pred_covariance = prediction

                        cutoff = 3 *  np.sqrt(
                            (pred_covariance[0,0] + pred_covariance[1,1])/2
                        )

                        line[line>cutoff] = 1e8

                        distance_matrix[ind_prediction] = line
                    ###

                    ### SOLVE LSA
                    _, inds_obs_mc = lsa(distance_matrix**2)
                    mc_preds.append(inds_obs_mc)
                    ###

                mc_preds = np.array(mc_preds)

                preds = np.tile(inds_obs_pre, (self.n_sampling,1))
                trackabilities = np.sum(preds == mc_preds, axis=0)/self.n_sampling

                trackability_scores.append(np.mean(trackabilities))
            ############## end trackability ###################



            ### "FAKE" UPDATE
            for ind_pred, ind_obs in zip(inds_pred, inds_obs):

                kalman_filter = states[ind_pred]
                matched_observation = observations_at_t[ind_obs]

                state, _ = kalman_filter.update(matched_observation)
                kalman_filter.state = np.array([
                    *matched_observation, *state[2:]
                ])

                states[ind_pred] = kalman_filter


                tracks[kalman_filter.index].append([ind_t, *matched_observation])
            ###


            ### MANAGING DISAPPEARANCES
            inds_disappearances = np.array(inds_predictions_at_t)[~np.isin(inds_predictions_at_t, inds_pred)]
            self.disappearances += len(inds_disappearances)

            states_copy = []
            for ind_state, state in enumerate(states):
                if not ind_state in inds_disappearances:

                    states_copy.append(state)
            
            states = states_copy
            ###


            ### MANAGING APPEARANCES
            inds_appearances = np.array(inds_observations_at_t)[~np.isin(inds_observations_at_t, inds_obs)]
            self.appearances += len(inds_appearances)

            for ind_to_appear in inds_appearances:

                new_observation = observations_at_t[ind_to_appear]

                new_kalman_filter = KalmanFilter(
                    transition_matrix=transition_matrix,
                    observation_matrix=observation_matrix,
                    transition_covariance= transition_covariance_matrix,
                    observation_covariance=observations_covariance_matrice,
                    index=index_track
                )

                new_kalman_filter.initialize_state(
                    state=np.array([*new_observation, 0, 0]),
                    state_covariance=states_initial_covariance
                )

                states.append(new_kalman_filter)

                tracks.append([[ind_t, *new_observation]])

                index_track += 1
            ###


            ### UPDATING KALMAN FILTERS WITH NEW dt
            for kalman_filter in states:
                # Change the transition matrix
                kalman_filter.F[0,2] = dt
                kalman_filter.F[1,3] = dt
            ###

        return tracks, trackability_scores