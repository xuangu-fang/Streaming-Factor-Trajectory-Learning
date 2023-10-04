"""
Implementation of Streaming Factor Trajectory for Dynamic Tensor, current is CP version, to be extended to Tucker 

The key differences of the idea and current one is: 
1. Build independent Trajectory Class (LDS-GP) for each embedding
2. Streaming update (one (batch) llk -> multi-msg to multi LDS -> filter_update simultaneously-> finally smooth back) 

Author: Shikai Fang
SLC, Utah, 2022.11
"""

import numpy as np
from numpy.lib import utils
import torch
import matplotlib.pyplot as plt
from model_LDS import LDS_GP_streaming
import os
import tqdm
import utils_streaming
import bisect
import tensorly as tl

tl.set_backend("pytorch")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
JITTER = 1e-4

torch.manual_seed(300)


class Streaming_Dynammic_Tensor_CP:

    def __init__(self, hyper_dict, data_dict):
        """-----------------hyper-paras---------------------"""
        self.device = hyper_dict["device"]
        self.R_U = hyper_dict["R_U"]  # rank of latent factor of embedding

        # prior of noise
        self.v = hyper_dict["v"]  # prior varience of embedding (scaler)
        self.a0 = hyper_dict["a0"]
        self.b0 = hyper_dict["b0"]
        self.DAMPING = hyper_dict["DAMPING"]
        self.DAMPING_tau = hyper_dict["DAMPING_tau"]

        self.product_method = "hadamard"  # CP
        """----------------data-dependent paras------------------"""
        # if kernel is matern-1.5, factor = 1, kernel is matern-2.5, factor =2

        self.ndims = data_dict["ndims"]
        self.nmods = len(self.ndims)

        self.tr_ind = data_dict["tr_ind"]
        self.tr_y = torch.tensor(data_dict["tr_y"]).to(self.device)  # N*1

        self.te_ind = data_dict["te_ind"]
        self.te_y = torch.tensor(data_dict["te_y"]).to(self.device)  # N*1

        self.train_time_ind = data_dict["tr_T_disct"]  # N_train*1
        self.test_time_ind = data_dict["te_T_disct"]  # N_test*1

        self.unique_train_time = list(np.unique(self.train_time_ind))

        self.time_uni = data_dict["time_uni"]  # N_time*1
        self.N_time = len(self.time_uni)

        LDS_streaming_paras = data_dict["LDS_streaming_paras"]

        # build dynamics (LDS-GP class) for each object in each mode (store using nested list)
        self.traj_class = []
        for mode in range(self.nmods):
            traj_class_mode = [
                LDS_GP_streaming(LDS_streaming_paras)
                for i in range(self.ndims[mode])
            ]
            self.traj_class.append(traj_class_mode)

        # posterior: store the most recently posterior from LDS for fast test?
        self.post_U_m = [
            torch.rand(dim, self.R_U, 1, self.N_time).double().to(self.device)
            for dim in self.ndims
        ]  #  (dim, R_U, 1, T) * nmod
        self.post_U_v = [
            torch.eye(self.R_U).reshape(
                (1, self.R_U, self.R_U,
                 1)).repeat(dim, 1, 1, self.N_time).double().to(self.device)
            for dim in self.ndims
        ]  # (dim, R_U, R_U, T) * nmod

        self.post_a = self.a0
        self.post_b = self.b0

        self.E_tau = 1

        # build time-data table: Given a time-stamp id, return the indexed of entries
        self.time_data_table_tr = utils_streaming.build_time_data_table(
            self.train_time_ind)

        self.time_data_table_te = utils_streaming.build_time_data_table(
            self.test_time_ind)

        # some place holders
        self.ind_T = None
        self.y_T = None
        self.uid_table = None
        self.data_table = None

        # store the msg in uid order
        self.msg_U_m = None
        self.msg_U_V = None

        # store the msg in data-llk order
        self.msg_U_lam_llk = None
        self.msg_U_eta_llk = None

        self.msg_a_llk = None
        self.msg_b_llk = None

        self.msg_gamma_lam = None
        self.msg_gamma_eta = None

        # gamma in CP, we just set it as a all-one constant v
        self.post_gamma_m = torch.ones(self.R_U,
                                       1).double().to(self.device)  # (R^K)*1

    def track_envloved_objects(self, T):
        """retrive the index/values/object-id of observed entries at T"""

        eind_T = self.time_data_table_tr[
            T]  # list of observed entries id at this time-stamp

        self.ind_T = self.tr_ind[eind_T]
        self.y_T = self.tr_y[eind_T].reshape(-1, 1, 1)

        self.N_T = len(self.y_T)

        self.uid_table, self.data_table = utils_streaming.build_id_key_table(
            nmod=self.nmods, ind=self.ind_T
        )  # nested-list of observed objects (and their associated entrie) at this time-stamp

    def filter_predict(self, T):
        """trajectories of involved objects take KF prediction step + update the posterior"""

        current_time_stamp = self.time_uni[T]

        for mode in range(self.nmods):
            for uid in self.uid_table[mode]:
                self.traj_class[mode][uid].filter_predict(current_time_stamp)

                # update the posterior based on the prediction state
                H = self.traj_class[mode][uid].H
                m = self.traj_class[mode][uid].m_pred_list[-1]
                P = self.traj_class[mode][uid].P_pred_list[-1]
                self.post_U_m[mode][uid, :, :, T] = torch.mm(H, m)
                self.post_U_v[mode][uid, :, :,
                                    T] = torch.mm(torch.mm(H, P), H.T)

    def product_with_gamma(self, E_z, E_z_2, mode):
        """product E_z / E_z_2 with gamma: for CP, gamma is constant all-one-vector, we actully do nothing here"""
        return E_z, E_z_2

    def msg_llk_init(self):
        """init the llk-msg used for DAMPING in inner loop of CEP, for CP, just msg for U and tau"""

        N_T = len(self.y_T)  # num of entries at current time-step

        # init the msg_U_llk, use natural parameters: lam = S_inv, eta = S_inv x m
        self.msg_U_lam_llk = [
            1e-4 * torch.eye(self.R_U).reshape((1, self.R_U, self.R_U)).repeat(
                N_T, 1, 1).double().to(self.device) for i in range(self.nmods)
        ]  # (N*R_U*R_U)*nmod
        self.msg_U_eta_llk = [
            1e-3 * torch.rand(N_T, self.R_U, 1).double().to(self.device)
            for i in range(self.nmods)
        ]  # (N*R_U*1)*nmod

        # msg of tau
        self.msg_a = torch.ones(N_T, 1).double().to(self.device)  # N*1
        self.msg_b = torch.ones(N_T, 1).double().to(self.device)  # N*1

    def msg_approx_U(self, T, mode):
        """approx the msg from the group of data-llk at T"""

        # reste msg_U_m, msg_U_V

        # self.msg_U_m = []
        # self.msg_U_V = []

        # for mode in range(self.nmods):
        msg_U_m_mode = []
        msg_U_V_mode = []

        condi_modes = [i for i in range(self.nmods)]
        condi_modes.remove(mode)  # [1,2] , [0,2]

        E_z, E_z_2 = utils_streaming.moment_product(
            modes=condi_modes,
            ind=self.ind_T,
            U_m=[ele[:, :, :, T] for ele in self.post_U_m],
            U_v=[ele[:, :, :, T] for ele in self.post_U_v],
            order="second",
            sum_2_scaler=False,
            device=self.device,
            product_method=self.product_method,
        )

        E_z, E_z_2 = self.product_with_gamma(E_z, E_z_2, mode)

        # use the nature-paras first, convinient to merge msg later
        msg_U_lam_new = self.E_tau * E_z_2  # (N,R,R)
        msg_U_eta_new = self.y_T * E_z * self.E_tau  # (N,R,1)

        # DAMPING step:
        self.msg_U_lam_llk[mode] = (self.DAMPING * self.msg_U_lam_llk[mode] +
                                    (1 - self.DAMPING) * msg_U_lam_new)

        self.msg_U_eta_llk[mode] = (self.DAMPING * self.msg_U_eta_llk[mode] +
                                    (1 - self.DAMPING) * msg_U_eta_new)

        # filling the msg_U_M, msg_U_V
        for i in range(len(self.uid_table[mode])):
            uid = self.uid_table[mode][i]  # id of embedding
            eid = self.data_table[mode][i]  # id of associated entries

            S_inv_cur = self.msg_U_lam_llk[mode][eid].sum(dim=0)  # (R,R)
            S_inv_Beta_cur = self.msg_U_eta_llk[mode][eid].sum(dim=0)  # (R,1)

            U_V = torch.linalg.inv(S_inv_cur)  # (R,R)
            U_M = torch.mm(U_V, S_inv_Beta_cur)  # (R,1)

            msg_U_m_mode.append(U_M)
            msg_U_V_mode.append(U_V)

        self.msg_U_m.append(msg_U_m_mode)
        self.msg_U_V.append(msg_U_V_mode)

    def msg_approx_tau(self, T):

        all_modes = [i for i in range(self.nmods)]

        E_z, E_z_2 = utils_streaming.moment_product(
            modes=all_modes,
            ind=self.ind_T,
            U_m=[ele[:, :, :, T] for ele in self.post_U_m],
            U_v=[ele[:, :, :, T] for ele in self.post_U_v],
            order="second",
            sum_2_scaler=False,
            device=self.device,
            product_method=self.product_method,
        )

        self.msg_a = 1.5 * torch.ones(self.N_T, 1).to(self.device)

        term1 = 0.5 * torch.square(self.y_T)  # N_T*1

        term2 = self.y_T.reshape(-1, 1) * torch.matmul(
            E_z.transpose(dim0=1, dim1=2), self.post_gamma_m).reshape(
                -1, 1)  # N_T*1

        temp = torch.matmul(E_z_2, self.post_gamma_m)  # N*R*1
        term3 = 0.5 * torch.matmul(temp.transpose(dim0=1, dim1=2),
                                   self.post_gamma_m).reshape(-1, 1)  # N*1

        # alternative way to compute term3, where we have to compute and store E_gamma_2
        # term3 = torch.unsqueeze(0.5* torch.einsum('bii->b',torch.bmm(self.E_gamma_2,self.E_z_2)),dim=-1) # N*1

        self.msg_b = self.DAMPING_tau * self.msg_b + (1 - self.DAMPING_tau) * (
            term1.reshape(-1, 1) - term2.reshape(-1, 1) + term3.reshape(-1, 1)
        )  # N*1

    def post_update_tau(self, T=None):
        """update post. factor of tau based on current msg. factors"""

        self.post_a = self.post_a + self.msg_a.sum() - self.N_T
        self.post_b = self.post_b + self.msg_b.sum()
        self.E_tau = self.post_a / self.post_b

    def filter_update(self, T, mode, add_to_list=True):
        """trajectories of involved objects take KF update step"""
        # for mode in range(self.nmods):
        for msg_id, uid in enumerate(self.uid_table[mode]):

            # we treat the approx msg as the observation values for KF
            y = self.msg_U_m[mode][msg_id]
            R = self.msg_U_V[mode][msg_id]

            # KF update step
            self.traj_class[mode][uid].filter_update(y=y,
                                                     R=R,
                                                     add_to_list=add_to_list)

            # update the posterior
            H = self.traj_class[mode][uid].H
            m = self.traj_class[mode][uid].m
            P = self.traj_class[mode][uid].P
            self.post_U_m[mode][uid, :, :, T] = torch.mm(H, m)
            self.post_U_v[mode][uid, :, :, T] = torch.mm(torch.mm(H, P), H.T)

    def smooth(self):
        """smooth back for all objects"""
        for mode in range(self.nmods):
            for uid in range(self.ndims[mode]):
                self.traj_class[mode][uid].smooth()

    def inner_smooth(self):
        """smooth back for online evaluation during the training, clean out the smooth-result after updating the the post_U"""

        self.smooth()
        self.get_post_U()

        for mode in range(self.nmods):
            for uid in range(self.ndims[mode]):
                self.traj_class[mode][uid].reset_smooth_list()

    def get_post_U(self):
        """get the final post of U using the smoothed result"""
        for T, time_stamp in enumerate(self.time_uni):
            for mode in range(self.nmods):
                for uid in range(self.ndims[mode]):
                    traj = self.traj_class[mode][uid]

                    if len(traj.time_stamp_list) > 0:
                        # at least have one observation

                        if time_stamp in traj.time_stamp_list:
                            # the time_stamp appread before

                            T_id = traj.time_2_ind_table[time_stamp]
                            # update the posterior based on the smoothed state

                            H = traj.H
                            m = traj.m_smooth_list[T_id]
                            P = traj.P_smooth_list[T_id]

                            self.post_U_m[mode][uid, :, :, T] = torch.mm(H, m)
                            self.post_U_v[mode][uid, :, :, T] = torch.mm(
                                torch.mm(H, P), H.T)

                        else:
                            # the time_stamp never appread before
                            # print(
                            #     "the time_stamp:", time_stamp, " never appread before"
                            # )

                            # locate the place of un-seen time_stamp
                            loc = bisect.bisect(traj.time_stamp_list,
                                                time_stamp)

                            if loc == 0:
                                # extrapolation at the first,backward Gaussian jump
                                prev_time_stamp = traj.time_stamp_list[loc]
                                prev_m = traj.m_smooth_list[loc]
                                prev_P = traj.P_smooth_list[loc]
                                prev_time_int = prev_time_stamp - time_stamp

                                prev_A = torch.inverse(
                                    torch.matrix_exp(traj.F *
                                                     prev_time_int).double())
                                prev_Q = traj.P_inf - torch.mm(
                                    torch.mm(prev_A, traj.P_inf), prev_A.T)

                                jump_m = torch.mm(prev_A, prev_m)
                                jump_P = (torch.mm(torch.mm(prev_A, prev_P),
                                                   prev_A.T) + prev_Q)

                                H = traj.H
                                self.post_U_m[mode][uid, :, :,
                                                    T] = torch.mm(H, jump_m)
                                self.post_U_v[mode][uid, :, :, T] = torch.mm(
                                    torch.mm(H, jump_P), H.T)

                            elif loc < len(traj.time_stamp_list):
                                # interpolation, merge (follow formulas 10-13 in draft)

                                prev_time_stamp = traj.time_stamp_list[loc - 1]
                                next_time_stamp = traj.time_stamp_list[loc]

                                prev_m = traj.m_smooth_list[loc - 1]
                                prev_P = traj.P_smooth_list[loc - 1]

                                next_m = traj.m_smooth_list[loc]
                                next_P = traj.P_smooth_list[loc]

                                prev_time_int = time_stamp - prev_time_stamp
                                next_time_int = next_time_stamp - time_stamp

                                prev_A = torch.matrix_exp(
                                    traj.F * prev_time_int).double()
                                prev_Q = traj.P_inf - torch.mm(
                                    torch.mm(prev_A, traj.P_inf), prev_A.T)

                                Q1_inv = torch.inverse(
                                    torch.mm(torch.mm(prev_A, prev_P),
                                             prev_A.T) + prev_Q)

                                next_A = torch.matrix_exp(
                                    traj.F * next_time_int).double()
                                next_Q = traj.P_inf - torch.mm(
                                    torch.mm(next_A, traj.P_inf), next_A.T)

                                Q2_inv = torch.inverse(
                                    torch.mm(torch.mm(next_A, next_P),
                                             next_A.T) + next_Q)

                                merge_P = torch.inverse(Q1_inv + torch.mm(
                                    next_A.T, torch.mm(Q2_inv, next_A)))

                                temp_term = torch.mm(
                                    Q1_inv, torch.mm(
                                        prev_A, prev_m)) + torch.mm(
                                            Q2_inv, torch.mm(next_A, next_m))
                                merge_m = torch.mm(merge_P, temp_term)

                                H = traj.H
                                self.post_U_m[mode][uid, :, :,
                                                    T] = torch.mm(H, merge_m)
                                self.post_U_v[mode][uid, :, :, T] = torch.mm(
                                    torch.mm(H, merge_P), H.T)

                            else:
                                # extrapolation at the end, foward gauss jump
                                prev_time_stamp = traj.time_stamp_list[loc - 1]
                                prev_m = traj.m_smooth_list[loc - 1]
                                prev_P = traj.P_smooth_list[loc - 1]
                                prev_time_int = time_stamp - prev_time_stamp

                                prev_A = torch.matrix_exp(
                                    traj.F * prev_time_int).double()
                                prev_Q = traj.P_inf - torch.mm(
                                    torch.mm(prev_A, traj.P_inf), prev_A.T)

                                jump_m = torch.mm(prev_A, prev_m)
                                jump_P = (torch.mm(torch.mm(prev_A, prev_P),
                                                   prev_A.T) + prev_Q)

                                H = traj.H
                                self.post_U_m[mode][uid, :, :,
                                                    T] = torch.mm(H, jump_m)
                                self.post_U_v[mode][uid, :, :, T] = torch.mm(
                                    torch.mm(H, jump_P), H.T)

    def model_test(self, test_ind, test_y, test_time):

        MSE_loss = torch.nn.MSELoss()
        MAE_loss = torch.nn.L1Loss()

        loss_test = {}

        all_modes = [i for i in range(self.nmods)]

        tid = test_time

        pred = utils_streaming.moment_product_T(
            modes=all_modes,
            ind=test_ind,
            ind_T=test_time,
            U_m_T=self.post_U_m,
            U_v_T=self.post_U_v,
            order="first",
            sum_2_scaler=True,
            device=self.device,
            product_method=self.product_method,
        )

        loss_test["rmse"] = torch.sqrt(
            MSE_loss(pred.squeeze(),
                     test_y.squeeze().to(self.device)))
        loss_test["MAE"] = MAE_loss(pred.squeeze(),
                                    test_y.squeeze().to(self.device))

        return pred, loss_test

    def reset(self):
        for mode in range(self.nmods):
            for uid in range(self.ndims[mode]):
                self.traj_class[mode][uid].reset_list()


class Streaming_Dynammic_Tensor_Tucker(Streaming_Dynammic_Tensor_CP):

    def __init__(self, hyper_dict, data_dict):
        super().__init__(hyper_dict, data_dict)

        self.DAMPING_gamma = hyper_dict["DAMPING_gamma"]

        self.product_method = "kronecker"
        self.nmod_list = [self.R_U for k in range(self.nmods)]
        """llk-msg and post. of vectorized Tucker-Core"""
        self.gamma_size = np.product([self.nmod_list])  # R_U^{K}

        # post. of gamma
        self.post_gamma_m = (torch.rand(self.gamma_size,
                                        1).double().to(self.device))  # (R^K)*1
        self.post_gamma_v = (torch.eye(self.gamma_size).double().to(
            self.device))  # (R^K)*(R^K)

    def product_with_gamma(self, E_z, E_z_2, mode):
        """product E_z / E_z_2 with gamma: for tucker, gamma is the folded tucker core, we actully do tensor-matrix product here"""

        E_gamma_tensor = tl.tensor(self.post_gamma_m.reshape(
            self.nmod_list))  # (R^k *1)-> (R * R * R ...)
        E_gamma_mat_k = tl.unfold(E_gamma_tensor, mode).double()

        # some mid terms (to compute E_a_2 = gamma_fold * z\z\.T *  gamma_fold.T)

        term1 = torch.matmul(E_z_2, E_gamma_mat_k.T)  # N * R_U^{K-1} * R_U
        E_a_2 = torch.matmul(term1.transpose(dim0=1, dim1=2),
                             E_gamma_mat_k.T).transpose(
                                 dim0=1, dim1=2)  # N * R_U * R_U

        # to compute E_a = gamma_fold * z\
        E_a = torch.matmul(E_z.transpose(dim0=1, dim1=2),
                           E_gamma_mat_k.T).transpose(
                               dim0=1, dim1=2)  # num_eid * R_U * 1

        return E_a, E_a_2

    def msg_llk_init(self):
        """init the llk-msg used for DAMPING in inner loop of CEP, for Tucker, include msg for U and tau and gamma"""

        N_T = len(self.y_T)  # num of entries at current time-step

        # init the msg_U_llk, use natural parameters: lam = S_inv, eta = S_inv x m
        self.msg_U_lam_llk = [
            1e-3 * torch.eye(self.R_U).reshape((1, self.R_U, self.R_U)).repeat(
                N_T, 1, 1).double().to(self.device) for i in range(self.nmods)
        ]  # (N*R_U*R_U)*nmod
        self.msg_U_eta_llk = [
            1e-3 * torch.rand(N_T, self.R_U, 1).double().to(self.device)
            for i in range(self.nmods)
        ]  # (N*R_U*1)*nmod

        # msg of tau
        self.msg_a = torch.ones(N_T, 1).double().to(self.device)  # N*1
        self.msg_b = torch.ones(N_T, 1).double().to(self.device)  # N*1

        # init msg of gamma
        self.msg_gamma_lam = 1e-4 * torch.eye(self.gamma_size).reshape(
            (1, self.gamma_size, self.gamma_size)).repeat(
                self.N_T, 1, 1).double().to(self.device)  # N*(R^K)*(R^K)
        self.msg_gamma_eta = 1e-4 * torch.rand(self.N_T, self.gamma_size,
                                               1).double().to(self.device)

    def msg_approx_gamma(self, T):

        all_modes = [i for i in range(self.nmods)]

        E_z, E_z_2 = utils_streaming.moment_product(
            modes=all_modes,
            ind=self.ind_T,
            U_m=[ele[:, :, :, T] for ele in self.post_U_m],
            U_v=[ele[:, :, :, T] for ele in self.post_U_v],
            order="second",
            sum_2_scaler=False,
            device=self.device,
            product_method=self.product_method,
        )

        msg_gamma_lam_new = self.E_tau * E_z_2  # N*(R^K)*(R^K)

        msg_gamma_eta_new = self.E_tau * E_z * self.y_T.reshape(-1, 1,
                                                                1)  # N*(R^K)*1

        self.msg_gamma_lam = (self.DAMPING_gamma * self.msg_gamma_lam +
                              (1 - self.DAMPING_gamma) * msg_gamma_lam_new
                              )  # N*(R^K)*(R^K)
        self.msg_gamma_eta = (self.DAMPING_gamma * self.msg_gamma_eta +
                              (1 - self.DAMPING_gamma) * msg_gamma_eta_new
                              )  # N*(R^K)*1

    def post_update_gamma(self, T=None):

        post_gamma_lam = torch.linalg.inv(self.post_gamma_v)
        post_gamma_eta = torch.mm(post_gamma_lam, self.post_gamma_m)

        self.post_gamma_v = torch.linalg.inv(
            self.msg_gamma_lam.sum(dim=0) + post_gamma_lam)  # (R^K) * (R^K)

        self.post_gamma_m = torch.mm(
            self.post_gamma_v, post_gamma_eta + self.msg_gamma_eta.sum(dim=0))

    def model_test(self, test_ind, test_y, test_time):

        MSE_loss = torch.nn.MSELoss()
        MAE_loss = torch.nn.L1Loss()

        loss_test = {}

        all_modes = [i for i in range(self.nmods)]

        E_z = utils_streaming.moment_product_T(
            modes=all_modes,
            ind=test_ind,
            ind_T=test_time,
            U_m_T=self.post_U_m,
            U_v_T=self.post_U_v,
            order="first",
            sum_2_scaler=False,
            device=self.device,
            product_method=self.product_method,
        )

        pred = torch.matmul(E_z.transpose(dim0=1, dim1=2),
                            self.post_gamma_m).squeeze()

        loss_test["rmse"] = torch.sqrt(
            MSE_loss(pred.squeeze(),
                     test_y.squeeze().to(self.device)))
        loss_test["MAE"] = MAE_loss(pred.squeeze(),
                                    test_y.squeeze().to(self.device))

        return pred, loss_test
