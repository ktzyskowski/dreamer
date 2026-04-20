from typing import Any, Union

import mlflow
import torch


MetricValue = Union[float, int, torch.Tensor]


class MetricsAggregator:
    """Wrapper around MLflow metric logging."""

    def __init__(
        self,
        experiment_name: str,
        log_every_n_gradient_steps: int = 10,
        log_system_metrics: bool = False,
    ):
        """Construct a new MLflow metrics aggregator.

        Args:
            experiment_name (str): MLflow experiment name
            log_every_n_gradient_steps (int): logging cadence.
            log_system_metrics (bool): whether to log system metrics (GPU utilization, etc.)
        """
        self.experiment_name = experiment_name
        self.log_every_n_gradient_steps = log_every_n_gradient_steps
        self.log_system_metrics = log_system_metrics

        self._pending: dict[str, MetricValue] = {}

    def update(self, metrics: dict[str, MetricValue]):
        """Update pending metrics, not logged until `maybe_flush` is called.

        Tensor values are detached but left on-device; they are only
        materialized to floats at flush time to avoid forcing a CPU sync
        on every gradient step.
        """
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
            self._pending[key] = value

    def maybe_flush(self, gradient_step: int):
        """Flush pending metrics, according to current gradient step and logging cadence."""
        if gradient_step % self.log_every_n_gradient_steps != 0:
            return
        if not self._pending:
            return
        resolved = {
            key: (value.item() if isinstance(value, torch.Tensor) else value)
            for key, value in self._pending.items()
        }
        mlflow.log_metrics(resolved, step=gradient_step)
        self._pending.clear()

    def log(self, metrics: dict[str, float], step: int):
        """Log metrics and immediately flush."""
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]):
        """Log experiment hyperparameters."""
        mlflow.log_params(params)

    def __enter__(self):
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(log_system_metrics=self.log_system_metrics)

    def __exit__(self, *_):
        mlflow.end_run()
