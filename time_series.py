""" Time Series Modeling module """

import os
from pathlib import Path
from datetime import datetime
import pickle
import logging

import numpy as np
import pandas as pd
from fbprophet import Prophet
from sklearn.model_selection import ParameterGrid

from utils import set_logger

set_logger()
logger = logging.getLogger("corona")

DEFAULT_PROPHET_PARAMS_GRID = {
    "growth": ["linear"],
    "seasonality_prior_scale": [1, 3, 10, 30],
    "changepoint_range": [0.8, 0.9],
    "changepoint_prior_scale": [0.01, 0.03, 0.1, 0.3, 1, 3],
    "n_changepoints": [10, 30],
    "seasonality_mode": ["additive", "multiplicative"],
}

class TSProphet(object):

    """
	This class generates time series forecasts using Facebook Prophet model for different set of options.
	The class can be used to make predictions from a pre-trained model, for training a model specifying
	the parameters and for finding the optimal parameters for the input series (tuning and training).

	Parameters
	----------
	input_series: time series input data
	input_freq: frequency of the time series
	validation_split_date: date for splitting training and validation
	train_until_date: end date of training period
	horizon_date: last date
	pretrained_model: str
	optimized_params: dict
	prophet_params_grid: dict
	add_country_holidays: bool
	add_black_friday: bool
	add_montly_seasonality: bool
	country_name: str
	transform: str
	save: bool
	file_name: str
	file_path: str
	use_pretrained_model: bool
	load_path_to_file: str

	Methods
	----------
	forecast

	"""

    def __init__(
        self,
        input_series: pd.Series,
        input_freq: str = "D",
        validation_split_date: str = None,
        train_until_date: str = None,
        horizon_date: str = None,
        optimized_params: dict = None,
        prophet_params_grid: dict = None,
        add_montly_seasonality: bool = False,
        transform: str = None,  # Allow logaritmic and Boxcox transformations
        save: bool = False,
        use_pretrained_model: bool = None,
        local_path: str = None,
        model_name: str = None
    ):
        assert isinstance(
            input_series.index, pd.DatetimeIndex
        ), "Index of the input must be 'DatetimeIndex'"

        if not input_series.index.is_monotonic:
            input_series.sort_index(inplace=True)
        self.input_series = input_series

        self.input_freq = input_freq

        # TODO: set default date for validation
        if validation_split_date is not None:
            self.validation_split_date = pd.Timestamp(validation_split_date)

        if prophet_params_grid is not None:
            self.prophet_params_grid = prophet_params_grid
        else:
            self.prophet_params_grid = DEFAULT_PROPHET_PARAMS_GRID

        # TODO: set default horizon
        if horizon_date is not None:
            self.horizon_date = pd.Timestamp(horizon_date)

        # TODO: set default (last date of series)
        if train_until_date is not None:
            self.train_until_date = pd.Timestamp(train_until_date)

        # TODO: review logic for using pre-trained model, tuning+training, training
        if use_pretrained_model is True:
            self.ts_model = self._load_model(local_path, model_name)
        else:
            if optimized_params is None:
                optimized_params = self._tune_parameters()

            self.ts_model = self._train_model(optimized_params, validation=False)
            if save is True:
                self._save_model(local_path, model_name, s3_save=s3_save)

    def _load_model(self, local_path, model_name):
        """
		Load pre-trained model. The model is loaded from a local location.

		Parameters
		----------
		local_path : str
			Path to pre-trained model.
		model_name: str
			Name of the pickle file.
		"""
        try:
            file = Path(local_path, model_name)
            if file.exists():
                with open(file, "rb") as f:
                    return pickle.load(f)
        except OSError:
            logger.error("Can't find model in local directory")

    def _eval_model(self, model):
        """
		Evaluate prophet model performance.

		Parameters
		----------
		model : Prophet object
			Model to evaluate
		"""
        y_true = self.input_series[
            self.validation_split_date : self.train_until_date
        ].rename("true")
        y_forecast = self._eval_forecast(model).rename("forecast")

        df_eval = pd.concat([y_true, y_forecast], axis=1).dropna()

        wmape = sum(np.abs(df_eval.true - df_eval.forecast)) / sum(df_eval.true) * 100

        return wmape

    def _train_eval_model(self, prophet_parameters):
        """
		Train and evaluate prophet model performance.

		Parameters
		----------
		prophet_parameters : dict
			Parameters compatible with Prophet model.
		"""
        model = self._train_model(prophet_parameters)
        acc = self._eval_model(model)

        return acc

    def _tune_parameters(self):
        """
		Finds the best parameters for prophet model on a validation set.
		"""
        grid = ParameterGrid(self.prophet_params_grid)
        # best_parameters = min(
        # 	grid,
        # 	key=lambda parameters: self._train_eval_model(prophet_parameters=parameters),
        # )
        acc_lst = []
        for i, p in enumerate(grid):
            print("Tuning parameters: {:2.1%}".format((i + 1) / len(grid)), end="\r")
            p["acc"] = self._train_eval_model(prophet_parameters=p)
            acc_lst.append(p)

        best_parameters = min(acc_lst, key=lambda x: x["acc"])

        del best_parameters["acc"]

        return best_parameters

    def _train_model(self, prophet_parameters, validation=True):
        """
		Train prophet model.

		Parameters
		----------
		prophet_parameters : dict
			Parameters compatible with Prophet model.
		validation : bool, default=True
			Whether to train with or without validation set.
		"""
        date_col = self.input_series.index.name
        val_col = self.input_series.name
        if validation:
            train_df = (
                self.input_series[: self.validation_split_date]
                .reset_index()
                .rename(columns={date_col: "ds", val_col: "y"})
            )
        else:
            train_df = (
                self.input_series[: self.train_until_date]
                .reset_index()
                .rename(columns={date_col: "ds", val_col: "y"})
            )

        # Disable prophet logging, see https://github.com/facebook/prophet/issues/223 for info
        with _suppress_stdout_stderr():
            if self.input_freq == "D":
                model = Prophet(
                    daily_seasonality=False,  # TODO: add option of monthly seasonality
                    yearly_seasonality=True,
                    weekly_seasonality=True,  # TODO: True if freq == 'D', False if freq == 'W'
                    **prophet_parameters,
                )
            elif self.input_freq[0] == "W":
                model = Prophet(
                    daily_seasonality=False,  # TODO: add option of monthly seasonality
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    **prophet_parameters,
                )

            model.fit(train_df)

            return model

    def _eval_forecast(self, model):
        """
		Train and evaluate prophet model performance.

		Parameters
		----------
		model : Prophet
			Model to use for forecasting
		"""

        date_col = self.input_series.index.name
        val_col = self.input_series.name

        if self.input_freq == "D":
            predict_range = (
                self.train_until_date - self.validation_split_date
            ).days + 1
        elif self.input_freq[0] == "W":
            predict_range = int(
                ((self.train_until_date - self.validation_split_date).days + 1) / 7
            )

        future = model.make_future_dataframe(
            periods=predict_range, freq=self.input_freq, include_history=False
        )

        forecast = model.predict(future)

        forecast_series = (
            forecast.assign(**{date_col: lambda x: pd.to_datetime(x.ds)})
            .rename(columns={"yhat": val_col})
            .set_index(date_col)[val_col]
        )

        return forecast_series

    def forecast(self):
        """
		Make predictions with trained model.
		"""
        date_col = self.input_series.index.name
        val_col = self.input_series.name

        if self.input_freq == "D":
            predict_range = (self.horizon_date - self.train_until_date).days + 1
        elif self.input_freq[0] == "W":
            predict_range = int(
                ((self.horizon_date - self.train_until_date).days + 1) / 7
            )

        future = self.ts_model.make_future_dataframe(
            periods=predict_range, freq=self.input_freq, include_history=False
        )

        forecast = self.ts_model.predict(future)

        # TODO: output also upper and lower forecasts
        forecast_series = (
            forecast.assign(**{date_col: lambda x: pd.to_datetime(x.ds)})
            .rename(columns={"yhat": val_col})
            .set_index(date_col)[val_col]
            .clip(lower=0)  # replacing negative predictions with 0
        )
        return forecast_series

    # TODO:
    def _add_country_holidays(self):
        pass

    def _save_model(self, local_path, model_name, s3_save):
        """
		Saves the model in a pickle file to a local directory or to the S3 bucket

		Parameters
		----------
		local_path : str
			Path to location where to save the model
		model_name : str
			Name of the file
		s3_save: bool
			Whether or not you want to save the model to the S3 bucket
		"""
        if s3_save:
            try:
                client = boto3.client("s3")
                serialized_object = pickle.dumps(self.ts_model)
                client.put_object(
                    Bucket="ngap--marketplace-analytics--prod--eu-west-1",
                    Key="qa/analytics/prophet-models/{}".format(model_name),
                    Body=serialized_object,
                )
                logger.info(
                    "Model saved in S3 bucket with filename: {}".format(model_name)
                )

            except ClientError as error:
                response = error.response.get("Error", dict()).get("Code", "")
                if response == "ExpiredToken":
                    logger.error(
                        "AWS Token has expired: run gimme-aws-creds to get the credentials"
                    )
                else:
                    logger.error("Unexpected AWS error: %s" % error)
                raise error
        else:
            path = Path(local_path)
            if path.is_dir() and model_name != None:
                file = Path(local_path, model_name)
                with open(file, "wb") as f:
                    pickle.dump(self.ts_model, f)
                logger.info("Model saved to local machine in %s" % file)
            else:
                logger.info("Path doesn't exist or model_name is not especified")
