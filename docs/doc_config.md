# Configuration
The configuration file is organized as follows:

- _data_:
    - _len_input_: length of the input series;
    - _horizon_forecast_: forecasting horizon (length of the output series);
    - _size_train_: fraction of data used for training;
    - _size_valid_: fraction of data used for validation;
- _model_:
    - _n_comp_trend_: power of the trend component;
    - _n_neur_hidden_: number of neurons in the hidden layer;
    - _frac_dropout_: fraction of dropout;
- _training_:
    - _batch_size_: batch size used for training;
    - _n_epochs_: number of epochs used for training;
    - _patience_: patience used for early stopping;
    - _min_delta_loss_perc_: minimum percentage required improvement of the validation loss; if its change is smaller than this value, then the patience counter increases.