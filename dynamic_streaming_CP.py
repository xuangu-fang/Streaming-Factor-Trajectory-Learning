from unittest import result
import numpy as np
import torch
from model_dynamic_streaming_tensor import Streaming_Dynammic_Tensor_CP
import utils_streaming
import tqdm
import yaml
import time

args = utils_streaming.parse_args_dynamic_streaming()

torch.random.manual_seed(args.seed)

# assert args.dataset in {'beijing_15k','beijing_20k', 'server', 'traffic'}

args.method = "CP"

print('dataset: ', args.dataset, ' rank: ', args.R_U)

config_path = "./config/config_" + args.dataset + "_" + args.method + ".yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

data_file = config["data_path"]
hyper_dict = utils_streaming.make_hyper_dict(config, args)

THRE = hyper_dict["THRE"]
INNER_ITER = hyper_dict["INNER_ITER"]

running_rmse = []
running_MAE = []
running_N = []
running_T = []
test_rmse = []
test_MAE = []

result_dict = {}
start_time = time.time()

for fold_id in range(args.num_fold):

    # running_rmse = []
    # running_MAE = []

    data_dict = utils_streaming.make_data_dict(hyper_dict, data_file, fold_id,
                                               args)

    model = Streaming_Dynammic_Tensor_CP(hyper_dict, data_dict)

    model.reset()

    N = 0

    for T_id in tqdm.tqdm(range(len(model.unique_train_time))):
        """ init_msg->filter_update->msg_approx->filter_update/post_update"""

        T = model.unique_train_time[T_id]
        model.track_envloved_objects(T_id)

        N = N + model.N_T

        model.filter_predict(T)
        model.msg_llk_init()

        for inner_it in range(INNER_ITER):

            old_post = utils_streaming.get_post(model, T)
            flag = (inner_it == (INNER_ITER - 1))

            model.msg_U_m = []
            model.msg_U_V = []

            if hyper_dict['CEP_UPDATE_INNNER_MODE'] == True:
                for mode in range(model.nmods):
                    model.msg_approx_U(T, mode)
                    model.filter_update(T, mode, flag)

            else:
                for mode in range(model.nmods):
                    model.msg_approx_U(T, mode)

                for mode in range(model.nmods):
                    model.filter_update(T, mode, flag)

            model.msg_approx_tau(T)
            model.post_update_tau(T)

            new_post = utils_streaming.get_post(model, T)

            relative_change = torch.square(new_post -
                                           old_post).sum() / old_post.norm()

            if flag:
                '''not converge till the MAX ITERATION'''
                pass
            elif relative_change < THRE:
                '''early converge'''

                if hyper_dict['CEP_UPDATE_INNNER_MODE'] == True:
                    for mode in range(model.nmods):
                        model.msg_approx_U(T, mode)
                        model.filter_update(T, mode, True)

                else:
                    for mode in range(model.nmods):
                        model.msg_approx_U(T, mode)

                    for mode in range(model.nmods):
                        model.filter_update(T, mode, True)

                model.msg_approx_tau(T)
                model.post_update_tau(T)

                break

        if hyper_dict["EVALU_T"] > 0 and fold_id == 0:
            "store the running test (only for the fold_0) "
            if T % hyper_dict["EVALU_T"] == 0:
                model.inner_smooth()
                _, test_result = model.model_test(model.te_ind, model.te_y,
                                                  model.test_time_ind)

                print("T:", T, "running_error", test_result['rmse'])
                running_MAE.append(test_result['MAE'].cpu().numpy().squeeze())
                running_rmse.append(
                    test_result['rmse'].cpu().numpy().squeeze())

                running_T.append(T)
                running_N.append(N)

    # pred, test_result = model.model_test(model.te_ind, model.te_y,
    #                                     model.test_time_ind)
    # print("test_error before smooth", test_result)

    model.smooth()
    model.get_post_U()

    pred, test_result = model.model_test(model.te_ind, model.te_y,
                                         model.test_time_ind)

    print("fold:", fold_id, "  test_error after smooth", test_result['rmse'])
    print("\n\n")

    test_MAE.append(test_result['MAE'].cpu().numpy().squeeze())
    test_rmse.append(test_result['rmse'].cpu().numpy().squeeze())

    if fold_id == 0:
        running_MAE.append(test_result['MAE'].cpu().numpy().squeeze())
        running_rmse.append(test_result['rmse'].cpu().numpy().squeeze())
        running_T.append(T)
        running_N.append(N)

rmse_array = np.array(test_rmse)
MAE_array = np.array(test_MAE)

running_rmse_array = np.array(running_rmse)
running_MAE_array = np.array(running_MAE)

result_dict['time'] = time.time() - start_time
result_dict['rmse_avg'] = rmse_array.mean()
result_dict['rmse_std'] = rmse_array.std()
result_dict['MAE_avg'] = MAE_array.mean()
result_dict['MAE_std'] = MAE_array.std()

result_dict['running_rmse'] = running_rmse_array
result_dict['running_MAE'] = running_MAE_array

result_dict['running_T'] = np.array(running_T)
result_dict['running_N'] = np.array(running_N)

utils_streaming.make_log(args, hyper_dict, result_dict)
