""" Time Series Modelling module """

import os
import pickle
import logging
import numpy as np
import pandas as pd
from fbprophet import Prophet  # TODO: Add to requirements.txt
from datetime import datetime

# import country_converter # TODO: Add to requirements.txt
from sklearn.model_selection import ParameterGrid
import boto3  # TODO: add to requirements.txt
from botocore.exceptions import ClientError
from pathlib import Path

logger = logging.getLogger("corona")

DEFAULT_PROPHET_PARAMS_GRID = {
    "growth": ["linear"],
    "seasonality_prior_scale": [1, 3, 10, 30],
    "changepoint_range": [0.8, 0.9],
    "changepoint_prior_scale": [0.01, 0.03, 0.1, 0.3, 1, 3],
    "n_changepoints": [5, 10, 30, 50],
    "seasonality_mode": ["additive"],
}

black_friday = pd.DataFrame(
    {
        "holiday": "black_friday",
        "ds": pd.to_datetime(["2018-11-23", "2019-11-29", "2020-11-11"]),
        #'lower_window': -6,
        #'upper_window': 6,
    }
)


class _suppress_stdout_stderr(object):
    """
	A context manager for doing a "deep suppression" of stdout and stderr in
	Python, i.e. will suppress all print, even if the print originates in a
	compiled C/Fortran sub-function.
	   This will not suppress raised exceptions, since exceptions are printed
	to stderr just before a script exits, and after the context manager has
	exited (at least, I think that is why it lets exceptions through).
	"""

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class TimeSeries(object):

    """
	This class generates time series forecasts using Facebook Prophet model for different set of options.
	The class can be used to make predictions from a pre-trained model, for training a model specifying
	the parameters and for finding the optimal parameters for the input series (tuning and training).

	Parameters
	----------
	input_series: pd.Series
	input_freq: str
	validation_split_date: str 
	train_until_date: str
	horizon_date: str
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
        add_country_holidays: bool = False,  # TODO
        add_black_friday: bool = False,
        add_montly_seasonality: bool = False,
        country_name: str = None,  # TODO
        transform: str = None,  # Allow logaritmic and Boxcox transformations
        save: bool = False,
        use_pretrained_model: bool = None,
        local_path: str = None,
        model_name: str = None,
        s3_load: bool = False,
        s3_save: bool = False,
    ):
        assert isinstance(
            input_series.index, pd.DatetimeIndex
        ), "Index of the input must be 'DatetimeIndex'"

        if not input_series.index.is_monotonic:
            input_series.sort_index(inplace=True)
        self.input_series = input_series

        self.input_freq = input_freq

        if validation_split_date is not None:
            self.validation_split_date = pd.Timestamp(validation_split_date)

        if prophet_params_grid is not None:
            self.prophet_params_grid = prophet_params_grid
        else:
            self.prophet_params_grid = DEFAULT_PROPHET_PARAMS_GRID

        if horizon_date is not None:
            self.horizon_date = pd.Timestamp(horizon_date)

        if train_until_date is not None:
            self.train_until_date = pd.Timestamp(train_until_date)

        # TODO: review logic for using pre-trained model, tuning+training, training
        if use_pretrained_model is True:
            self.ts_model = self._load_model(local_path, model_name, s3_load=s3_load)
        else:
            if optimized_params is None:
                optimized_params = self._tune_parameters()

            self.ts_model = self._train_model(optimized_params, validation=False)
            if save is True:
                self._save_model(local_path, model_name, s3_save=s3_save)

    def _load_model(self, local_path, model_name, s3_load):
        """
		Load pre-trained model. The model can be load either from a local location or from the S3 bucket.

		Parameters
		----------
		local_path : str
			Path to pre-trained model.
		model_name: str
			Name of the pickle file.
		s3_load: bool
			Whether or not you want to load the model from the S3 bucket
		"""
        if s3_load:
            try:
                client = boto3.client("s3")

                response = client.get_object(
                    Bucket="ngap--marketplace-analytics--prod--eu-west-1",
                    Key="qa/analytics/prophet-models/{name}".format(name=model_name),
                )

                serialized_object = response["Body"].read()

                return pickle.loads(serialized_object)
                logger.info("{} Prophet model loaded from S3 bucket".format(model_name))

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
            file = Path(local_path, model_name)
            if file.exists():
                with open(file, "rb") as f:
                    return pickle.load(f)

        raise OSError("Can't find model in local directory nor in AWS")

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
