import numpy as np
from pathlib import Path


def process_to_txt(dataset_name):

    dict_name = '../for_matlab/' + dataset_name + '/'
    Path(dict_name).mkdir(parents=True, exist_ok=True)
    data_file = '../' + dataset_name + '.npy'
    full_data = np.load(data_file, allow_pickle=True).item()

    for fold in range(5):

        data_dict = full_data['data'][fold]
        data_dict['ndims'] = full_data['ndims']
        data_dict['ndims'] = full_data['ndims'] + [len(full_data['time_uni'])]

        data_dict['tr_ind'] = np.concatenate(
            [data_dict['tr_ind'], data_dict['tr_T_disct'].reshape(-1, 1)], 1)
        data_dict['te_ind'] = np.concatenate(
            [data_dict['te_ind'], data_dict['te_T_disct'].reshape(-1, 1)], 1)

        train_data = np.concatenate(
            [data_dict['tr_ind'], data_dict['tr_y'].reshape(-1, 1)], 1)
        test_data = np.concatenate(
            [data_dict['te_ind'], data_dict['te_y'].reshape(-1, 1)], 1)

        fmt = ['%d' for i in range(len(data_dict['ndims']))] + ['%.3f']

        file_train = dict_name + dataset_name + "_train_" + str(fold) + '.txt'
        file_test = dict_name + dataset_name + "_test_" + str(fold) + '.txt'

        np.savetxt(file_train, train_data, fmt=fmt, delimiter=' ')
        np.savetxt(file_test, test_data, fmt=fmt, delimiter=' ')

    ndims_info = 'dataset: %s, ndims: %s' % (dataset_name,
                                             str(data_dict['ndims']))

    f = open(dict_name + 'ndims.txt', "w+")
    f.write(ndims_info)
    f.write(
        "\n  base-0 indexing, space-separated, last-column is entry values, last-mode is DDT-mode (use for static baseline, drop for streaming baselines), all entries ordered by DDT-mode (last mode)"
    )


# dataset_name = 'beijing_15k'
# dataset_name = 'beijing_20k'
# dataset_name = 'traffic_48k'
# dataset_name = 'server_10k'

# process_to_txt('beijing_15k')
# process_to_txt('beijing_20k')
# process_to_txt('traffic_48k')
# process_to_txt('server_10k')
process_to_txt('fitRecord_50k')
