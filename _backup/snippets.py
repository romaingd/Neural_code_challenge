###############################################################################
#                                                                             #
#                          tsfresh features computing                         #
#                                                                             #
###############################################################################

if recompute_training:
    print('Processing training set...')
    split_breaks = [int(n_tr / nb_splits) * i for i in range(nb_splits)] + [n_tr]
    for i in range(nb_splits):
        start = split_breaks[i]
        stop = split_breaks[i + 1]
        print('Number of rows processed:', stop - start)
        features_tr = extract_features(TSFormatting().transform(x_tr.iloc[start:stop]),
                                        column_id='id', column_sort='time',
                                        default_fc_parameters=EfficientFCParameters())
        features_tr['neuron_id'] = x_tr.iloc[start:stop]['neuron_id']
        if (i == 0):
            features_tr.to_csv(features_folder + 'feat_tr.csv', mode='w', header=True, index=False)
        else:
            features_tr.to_csv(features_folder + 'feat_tr.csv', mode='a', header=False, index=False)
        del features_tr

if recompute_test:
    print('Processing test set...')
    split_breaks = [int(n_te / nb_splits) * i for i in range(nb_splits)] + [n_te]
    for i in range(nb_splits):
        start = split_breaks[i]
        stop = split_breaks[i + 1]
        print('Number of rows processed:', stop - start)
        features_te = extract_features(TSFormatting().transform(x_te.iloc[start:stop]),
                                        column_id='id', column_sort='time',
                                        default_fc_parameters=EfficientFCParameters())
        features_te['neuron_id'] = x_te.iloc[start:stop]['neuron_id']
        if (i == 0):
            features_te.to_csv(features_folder + 'feat_te.csv', mode='w', header=True, index=False)
        else:
            features_te.to_csv(features_folder + 'feat_te.csv', mode='a', header=False, index=False)
        del features_te


