"""
Flower strategy that puts robust.py between the clients and the average.

robust.py screens updates and the validation gate refuses bad candidates,
but until this module nothing called it: every aggregator ran plain
FedAvg, so the defence was tested and disconnected. This is the
connection.

    RobustFedAvg(aggregation="robust")   screen, then average survivors
    RobustFedAvg(aggregation="fedavg")   plain FedAvg, the baseline arm

Keeping FedAvg behind the same class is deliberate. The comparison study
needs both arms run by identical server code, differing only in this one
switch, or a difference in results could come from the plumbing.

The screening works on deltas against the current global model, so the
strategy remembers the parameters it last sent out (`configure_fit`
receives them every round). When everything is rejected it returns no
parameters, and Flower keeps the current global model rather than
promoting anything.

Imported two ways: as `src.federated.robust_strategy` from the repo, and
as a flat module next to `robust.py` inside the central-server container,
which ships without the rest of `src/`.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

try:
    from src.federated.robust import AggregationReport, ClientUpdate, RobustAggregator
except ImportError:  # flat copy in the central-server container
    from robust import AggregationReport, ClientUpdate, RobustAggregator

logger = logging.getLogger(__name__)

AGGREGATIONS = ("robust", "fedavg")


class RobustFedAvg(FedAvg):
    """FedAvg whose `aggregate_fit` screens updates through RobustAggregator."""

    def __init__(
        self,
        *,
        aggregation: str = "robust",
        aggregator: Optional[RobustAggregator] = None,
        on_report: Optional[Callable[[int, AggregationReport], None]] = None,
        **kwargs,
    ):
        if aggregation not in AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}"
            )
        super().__init__(**kwargs)
        self.aggregation = aggregation
        self.aggregator = aggregator or RobustAggregator()
        self.on_report = on_report
        self.last_report: Optional[AggregationReport] = None
        self._global: Optional[List] = None
        if kwargs.get("initial_parameters") is not None:
            self._global = parameters_to_ndarrays(kwargs["initial_parameters"])

    def __repr__(self) -> str:
        return f"RobustFedAvg(aggregation={self.aggregation})"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        # The parameters going out this round are the reference every
        # returned update is measured against.
        self._global = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if self.aggregation == "fedavg":
            return super().aggregate_fit(server_round, results, failures)

        if not results:
            return None, {}
        if failures and not self.accept_failures:
            return None, {}
        if self._global is None:
            # Flower always calls configure_fit first, so this means the
            # strategy is being driven by something other than a server.
            # Without a reference there are no deltas to screen.
            logger.warning(
                "Round %d: no global model to screen against; falling back to FedAvg",
                server_round,
            )
            return super().aggregate_fit(server_round, results, failures)

        ids = _unique_ids(results)
        updates = [
            ClientUpdate(
                node_id=nid,
                weights=parameters_to_ndarrays(res.parameters),
                num_samples=int(res.num_examples),
                metrics=dict(res.metrics or {}),
            )
            for nid, (proxy, res) in zip(ids, results)
        ]
        new_weights, report = self.aggregator.aggregate(self._global, updates)
        self.last_report = report

        for s in report.screenings:
            if not s.accepted:
                logger.warning("Round %d: rejected %s", server_round, s)
        logger.info(
            "Round %d robust aggregation: %d accepted, %d rejected, consensus=%s. %s",
            server_round, len(report.accepted), len(report.rejected),
            report.consensus_applied, report.note,
        )

        if self.on_report is not None:
            try:
                self.on_report(server_round, report)
            except Exception as exc:  # reporting must never break a round
                logger.warning("on_report failed: %s", exc)

        # Client metrics are averaged over the updates that were kept, so a
        # rejected node's numbers do not describe a model it did not shape.
        kept = set(report.accepted)
        metrics: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (res.num_examples, res.metrics)
                for nid, (proxy, res) in zip(ids, results)
                if nid in kept
            ]
            if fit_metrics:
                metrics = dict(self.fit_metrics_aggregation_fn(fit_metrics))
        metrics.update({
            "robust_accepted": len(report.accepted),
            "robust_rejected": len(report.rejected),
            "robust_consensus": int(report.consensus_applied),
            "robust_report": json.dumps(report.as_dict()),
        })

        if new_weights is None:
            logger.warning(
                "Round %d: no update survived screening (%s); keeping the current model",
                server_round, report.note,
            )
            return None, metrics
        return ndarrays_to_parameters(new_weights), metrics


def _node_id(proxy: ClientProxy, res: FitRes) -> str:
    """The camera's own name when it sent one, else Flower's connection id."""
    named = (res.metrics or {}).get("node_id")
    return str(named or proxy.cid)


def _unique_ids(results: List[Tuple[ClientProxy, FitRes]]) -> List[str]:
    """One id per update. RobustAggregator keys deltas by node id, so two
    nodes left on the same default name would otherwise overwrite each
    other and one update would be screened twice while the other vanished."""
    seen: Dict[str, int] = {}
    out = []
    for proxy, res in results:
        nid = _node_id(proxy, res)
        if nid in seen:
            seen[nid] += 1
            nid = f"{nid}#{seen[nid]}"
        else:
            seen[nid] = 0
        out.append(nid)
    return out
