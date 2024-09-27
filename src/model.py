import numpy as np
import torch
import sklearn

class Block(torch.nn.Module):
    def __init__(self, dict_params: dict, component: str, num_features: int) -> None:
        '''
        Block of the N-BEATS architecture.
        
        Args:
            dict_params: Dictionary containing the configuration settings.
            component: String representing the component to be modelled; it can be either 'trend' or 'seasonality'.
            num_features: Number of features of the input series.
            
        Returns: None
        '''
        super().__init__()
        #
        len_input = dict_params['data']['len_input']
        horizon_forecast = dict_params['data']['horizon_forecast']
        n_comp_trend = dict_params['model']['n_comp_trend']
        n_neur_hidden = dict_params['model']['n_neur_hidden']
        frac_dropout = dict_params['model']['frac_dropout']
        self.component = component
        self.horizon_forecast = horizon_forecast
        self.n_comp_trend = n_comp_trend
        # hidden layers of the FC stack
        self.dense_hid_1 = torch.nn.Linear(in_features = len_input*num_features, out_features = n_neur_hidden)
        self.dense_hid_2 = torch.nn.Linear(in_features = n_neur_hidden, out_features = n_neur_hidden)
        self.dense_hid_3 = torch.nn.Linear(in_features = n_neur_hidden, out_features = n_neur_hidden)
        self.dense_hid_4 = torch.nn.Linear(in_features = n_neur_hidden, out_features = n_neur_hidden)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p = frac_dropout)
        self.batch_norm_1 = torch.nn.BatchNorm1d(n_neur_hidden)
        self.batch_norm_2 = torch.nn.BatchNorm1d(n_neur_hidden)
        self.batch_norm_3 = torch.nn.BatchNorm1d(n_neur_hidden)
        self.batch_norm_4 = torch.nn.BatchNorm1d(n_neur_hidden)
        # dense layer for producing the theta's
        self.dense_theta_b = torch.nn.Linear(in_features = n_neur_hidden, out_features = n_neur_hidden, bias = False)
        if component == 'trend':
            self.dense_theta_f = torch.nn.Linear(in_features = n_neur_hidden, out_features = self.n_comp_trend + 1, bias = False)
        if component == 'seasonality':
            self.dense_theta_f = torch.nn.Linear(in_features = n_neur_hidden, out_features = 2*int(np.floor(horizon_forecast/2-1)) + 1, bias = False)
        # trend and seasonality matrices
        self.mat_T = np.hstack([(np.arange(horizon_forecast).reshape(-1, 1)/horizon_forecast)**p for p in range(n_comp_trend+1)]).astype(np.float32)
        self.mat_S = np.hstack([np.arange(horizon_forecast).reshape(-1, 1)**0] +
                               [np.cos(2*np.pi*i*np.arange(horizon_forecast).reshape(-1, 1)/horizon_forecast) for i in
                                range(1, int(np.floor(horizon_forecast/2-1)) + 1)] +
                               [np.sin(2*np.pi*i*np.arange(horizon_forecast).reshape(-1, 1)/horizon_forecast) for i in
                                range(1, int(np.floor(horizon_forecast/2-1)) + 1)]).astype(np.float32)
        self.mat_T, self.mat_S = torch.tensor(self.mat_T), torch.tensor(self.mat_S)
        # final layer for backcasting
        self.dense_x_hat = torch.nn.Linear(in_features = n_neur_hidden, out_features = len_input)
        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        #
        y = self.dense_hid_1(x)
        y = self.batch_norm_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        for i in range(3):
            hidden_layer = [self.dense_hid_2, self.dense_hid_3, self.dense_hid_4][i]
            batch_norm = [self.batch_norm_2, self.batch_norm_3, self.batch_norm_4][i]
            y = hidden_layer(y)
            y = batch_norm(y)
            y = self.relu(y)
            y = self.dropout(y)
        # compute theta's
        theta_b = self.dense_theta_b(y)
        theta_f = self.dense_theta_f(y)
        # compute backcast
        x_hat = self.dense_x_hat(theta_b)
        x_hat = x_hat.reshape(*x_hat.shape, 1)
        # compute time series components
        if self.component == 'trend':
            y_hat = torch.matmul(self.mat_T.repeat(theta_f.shape[0], 1, 1), theta_f.reshape(*theta_f.shape, 1))
        if self.component == 'seasonality':
            y_hat = torch.matmul(self.mat_S.repeat(theta_f.shape[0], 1, 1), theta_f.reshape(*theta_f.shape, 1))
        #
        return x_hat, y_hat

class Stack(torch.nn.Module):
    def __init__(self, dict_params: dict, component: str, num_features: int) -> None:
        '''
        Stack of the N-BEATS architecture.
        
        Args:
            dict_params: Dictionary containing the configuration settings.
            component: String representing the component to be modelled; it can be either 'trend' or 'seasonality'.
            num_features: Number of features of the input series.
            
        Returns: None
        '''
        super().__init__()
        #
        self.block_1 = Block(dict_params = dict_params, component = component, num_features = num_features)
        self.block_2 = Block(dict_params = dict_params, component = component, num_features = num_features)
        self.block_3 = Block(dict_params = dict_params, component = component, num_features = num_features)
        
    def forward(self, x):
        # first block
        x_hat, y = self.block_1(x)
        x = x - x_hat
        y_hat = y
        # second block
        x_hat, y = self.block_2(x)
        x = x - x_hat
        y_hat += y
        # third block
        x_hat, y = self.block_3(x)
        x = x - x_hat
        y_hat += y
        #
        x_hat = x
        return x_hat, y_hat

class NBeats(torch.nn.Module):
    def __init__(self, dict_params: dict, num_features: int) -> None:
        '''
        N-BEATS architecture.
        
        Args:
            dict_params: Dictionary containing the configuration settings.
            num_features: Number of features of the input series.
            
        Returns: None
        '''
        super().__init__()
        #
        self.stack_trend = Stack(dict_params = dict_params, component = 'trend', num_features = num_features)
        self.stack_seas = Stack(dict_params = dict_params, component = 'seasonality', num_features = num_features)
        
    def forward(self, x):
        # trend stack
        x_hat, y_hat_trend = self.stack_trend(x)
        x = x - x_hat
        # seasonality stack
        _, y_hat_seas = self.stack_seas(x)
        #
        return y_hat_trend, y_hat_seas
    
class TrainNBeats:
    def __init__(self, model: torch.nn.Module, dict_params: dict, dataloader_train: torch.utils.data.DataLoader, dataloader_valid: torch.utils.data.DataLoader):
        '''
        Class to train the N-BEATS model.
        
        Args:
            model: PyTorch model.
            dict_params: Dictionary containing information about the model architecture.
            dataloader_train: Dataloader containing training data.
            dataloader_valid: Dataloader containing validation data.
            
        Returns: None.
        '''
        self.model = model
        self.dict_params = dict_params
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params = model.parameters())
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer, factor = 0.5)
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid

    def _model_on_batch(self, batch: tuple, training: bool, loss_epoch: float) -> float:
        '''
        Function to perform training on a single batch of data.
        
        Args:
            batch: Batch of data to use for training/evaluation.
            training: Whether to perform training (if not, evaluation is understood).
            loss_epoch: Loss of the current epoch.
            
        Returns:
            loss: Value of the loss function.
        '''
        if training == True:
            self.optimizer.zero_grad()
        #
        x, y_true = batch
        x = x.to('cpu')
        y_true = y_true.to('cpu')
        y_hat_trend, y_hat_seas = self.model(x)
        y_hat_trend = y_hat_trend.to('cpu')
        y_hat_seas = y_hat_seas.to('cpu')
        #
        loss = self.loss_func(y_hat_trend + y_hat_seas, y_true)
        #
        if training == True:
            loss.backward()
            self.optimizer.step()
        #
        return loss.item()

    def _train(self) -> float:
        '''
        Function to train the N-BEATS model on a single epoch.
        
        Args: None.
            
        Returns:
            loss: Value of the training loss function per batch.
        '''
        self.model.train()
        loss_epoch = 0
        for batch in self.dataloader_train:
            loss_epoch += self._model_on_batch(batch = batch, training = True, loss_epoch = loss_epoch)
        return loss_epoch/len(self.dataloader_train)

    def _eval(self) -> float:
        '''
        Function to evaluate the N-BEATS model on the validation set on a single epoch.
        
        Args: None.
            
        Returns:
            loss: Value of the validation loss function per batch.
        '''
        self.model.eval()
        loss_epoch = 0
        with torch.no_grad():
            for batch in self.dataloader_valid:
                loss_epoch += self._model_on_batch(batch = batch, training = False, loss_epoch = loss_epoch)
        return loss_epoch/len(self.dataloader_valid)

    def train_model(self) -> (torch.nn.Module, list, list):
        '''
        Function to train the N-BEATS model.
        
        Args: None.
            
        Returns:
            model: Trained N-BEATS model.
            list_loss_train: List of training loss function across the epochs.
            list_loss_valid: List of validation loss function across the epochs.
        '''
        dict_params = self.dict_params
        n_epochs = dict_params['training']['n_epochs']
        list_loss_train, list_loss_valid = [], []
        counter_patience = 0
        for epoch in range(1, n_epochs + 1):
            loss_train = self._train()
            loss_valid = self._eval()
            #
            if (len(list_loss_valid) > 0) and (loss_valid >= np.min(list_loss_valid)*(1 - dict_params['training']['min_delta_loss_perc'])):
                counter_patience += 1
            if (len(list_loss_valid) == 0) or ((len(list_loss_valid) > 0) and (loss_valid < np.min(list_loss_valid))):
                counter_patience = 0
                torch.save(self.model.state_dict(), '../data/artifacts/weights.p')
            if counter_patience >= dict_params['training']['patience']:
                print(f'Training stopped at epoch {epoch}. Restoring weights from epoch {np.argmin(list_loss_valid) + 1}.')
                self.model.load_state_dict(torch.load('../data/artifacts/weights.p'))
                break
            #
            print(f'Epoch {epoch}: training loss = {loss_train:.4f}, validation loss = {loss_valid:.4f}, patience counter = {counter_patience}.')
            self.scheduler.step(loss_valid)
            #
            list_loss_train.append(loss_train)
            list_loss_valid.append(loss_valid)
        if epoch == n_epochs:
            self.model.load_state_dict(torch.load('../data/artifacts/weights.p'))
        return self.model, list_loss_train, list_loss_valid
    
def get_y_true_y_hat(model: torch.nn.Module, x: torch.tensor, y: torch.tensor, date_y: np.array,
                     scaler: sklearn.preprocessing.StandardScaler) -> (np.array, np.array, np.array):
    '''
    Function to get the real time series and its prediction.
    
    Args:
        model: Trained N-BEATS model.
        x: Tensor representing regressors.
        y: Tensor representing target time series.
        date_y: Array containing the dates corresponding to the elements of `y`.
        scaler: Scaled used to rescale data.
        
    Returns:
        y_true: Array containing the true values.
        y_hat_trend: Array containing the predicted trend.
        y_hat_seas: Array containing the predicted seasonality.
    '''
    list_date = []
    y_true = []
    y_hat_trend, y_hat_seas = [], []
    pred_trend, pred_seas = model(x)
    for i in range(np.unique(date_y).shape[0]):
        date = np.unique(date_y)[i]
        list_date.append(date)
        idx = np.where(date_y == date)
        y_true.append(y.numpy()[idx].mean())
        y_hat_trend.append(pred_trend.detach().numpy()[idx].mean())
        y_hat_seas.append(pred_seas.detach().numpy()[idx].mean())
    y_true = np.array(y_true)
    y_hat_trend = np.array(y_hat_trend)
    y_hat_seas = np.array(y_hat_seas)
    # scale back
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_hat_trend = scaler.inverse_transform(y_hat_trend.reshape(-1, 1)).ravel()
    y_hat_seas = scaler.inverse_transform(y_hat_seas.reshape(-1, 1)).ravel()
    #
    return y_true, y_hat_trend, y_hat_seas

def compute_mape(y_true: np.array, y_hat_trend: np.array, y_hat_seas: np.array, scaler: sklearn.preprocessing.StandardScaler) -> float:
    '''
    Function to compute the MAPE.
    
    Args:
        y_true: Array containing the true values.
        y_hat_trend: Array containing the predicted trend.
        y_hat_seas: Array containing the predicted seasonality.
        scaler: Scaled used to rescale data.
        
    Returns:
        mape: MAPE computed from `y_true` and `y_hat`.
    '''
    y_hat = y_hat_trend + y_hat_seas - scaler.mean_
    mape = np.mean(abs(y_true[y_true > 0] - y_hat[y_true > 0])/y_true[y_true > 0])
    return mape