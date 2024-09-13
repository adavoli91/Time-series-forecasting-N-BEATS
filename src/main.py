import pandas as pd
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader
from preprocessing import *
from model import *

def main():
    dict_params = yaml.load(open('../config.yaml'), Loader = yaml.FullLoader)
    #
    df = pd.read_csv('../data/raw/train.csv', parse_dates = ['date'], index_col = 'id')
    df = df.groupby(['date', 'family']).agg({'sales': 'sum'}).reset_index()
    # add all dates
    df_temp = []
    for family in df['family'].unique():
        df_temp.append(pd.DataFrame({'date': pd.date_range(df['date'].min(), df['date'].max())}).merge(df[df['family'] == family],
                                                                                                    on = 'date', how = 'left'))
        df_temp[-1]['family'] = family
        df_temp[-1]['sales'] = df_temp[-1]['sales'].ffill()
    df = pd.concat(df_temp).reset_index(drop = True)
    del df_temp
    df = df.rename(columns = {'sales': 'y'})
    # perform train-test splitting
    df_train, df_valid, df_test, scaler = train_test_splitting(df = df[df['family'] == 'AUTOMOTIVE'].reset_index(drop = True).drop('family', axis = 1),
                                                            dict_params = dict_params)
    #
    horizon_forecast = dict_params['data']['horizon_forecast']
    # training set data
    x_train, y_train, date_x_train, date_y_train = get_x_y(df = df_train, df_future = df_valid, dict_params = dict_params,
                                                        test_set = False, horizon_forecast = horizon_forecast)
    # validation set data
    x_valid, y_valid, date_x_valid, date_y_valid = get_x_y(df = df_valid, df_future = df_test, dict_params = dict_params,
                                                        test_set = False, horizon_forecast = horizon_forecast)
    # test set data
    x_test, y_test, date_x_test, date_y_test = get_x_y(df = df_test, df_future = None, dict_params = dict_params, test_set = True,
                                                    horizon_forecast = horizon_forecast)
    # create datasets and dataloader
    dataset_train = CreateDataset(x = x_train, y = y_train)
    dataset_valid = CreateDataset(x = x_valid, y = y_valid)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = dict_params['training']['batch_size'], shuffle = True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size = dict_params['training']['batch_size'], shuffle = False)
    # train model
    model = NBeats(dict_params = dict_params)
    model, list_loss_train, list_loss_valid = TrainNBeats(model = model, dict_params = dict_params, dataloader_train = dataloader_train, dataloader_valid = dataloader_train).train_model()
    # evaluate results
    model = NBeats(dict_params = dict_params)
    model.load_state_dict(torch.load('../data/artifacts/weights.p'))
    model.eval()
    # get time series and the corresponding predictions
    y_true_train, y_hat_trend_train, y_hat_seas_train = get_y_true_y_hat(model = model, x = x_train, y = y_train, date_y = date_y_train, scaler = scaler)
    y_true_valid, y_hat_trend_valid, y_hat_seas_valid = get_y_true_y_hat(model = model, x = x_valid, y = y_valid, date_y = date_y_valid, scaler = scaler)
    y_true_test, y_hat_trend_test, y_hat_seas_test = get_y_true_y_hat(model = model, x = x_test, y = y_test, date_y = date_y_test, scaler = scaler)
    # compute mape on training, validation and test set
    mape_train = compute_mape(y_true = y_true_train, y_hat_trend = y_hat_trend_train, y_hat_seas = y_hat_seas_train, scaler = scaler)
    mape_valid = compute_mape(y_true = y_true_valid, y_hat_trend = y_hat_trend_valid, y_hat_seas = y_hat_seas_valid, scaler = scaler)
    mape_test = compute_mape(y_true = y_true_test, y_hat_trend = y_hat_trend_test, y_hat_seas = y_hat_seas_test, scaler = scaler)
    #
    plt.figure(figsize = [10, 6])
    plt.plot(np.unique(date_y_test), y_true_test, label = 'True', color = 'r')
    plt.plot(np.unique(date_y_test), y_hat_trend_test + y_hat_seas_test - scaler.mean_, label = 'Predicted', color = 'b')
    plt.xlabel('Date', fontsize = 16)
    plt.ylabel('Sales', fontsize = 16)
    plt.xticks(rotation = 45)
    plt.legend()
    plt.savefig('../docs/figures_for_readme/result', bbox_inches = 'tight')

if __name__ == '__main__':
    df_result = main()