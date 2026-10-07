from __future__ import annotations

import math

import numpy as np

from .evaluation import evaluate_global_method, evaluate_local_fcm, evaluate_personalized_method
from .fcm import fcm, initialize_centers_from_data
from .federated import (
    ditto_fcm,
    federated_fcm,
    federated_fcm_model_averaging,
    fedprox_fcm,
    ifca_fcm,
    personalized_federated_fcm,
)
from .synthetic import ClientData


def relevance_metrics(relevance: np.ndarray | None, low_threshold: float = 0.2) -> dict[str, float]:
    if relevance is None:
        return {}
    values = np.asarray(relevance, dtype=float)
    return {
        "mean_relevance": float(np.mean(values)),
        "std_relevance": float(np.std(values)),
        "min_observed_relevance": float(np.min(values)),
        "max_observed_relevance": float(np.max(values)),
        "low_relevance_share": float(np.mean(values <= low_threshold)),
    }


def add_result_row(
    rows: list[dict[str, object]],
    context: dict[str, object],
    method: str,
    metrics: dict[str, float],
    objective_last: float | None,
    rounds: int | None,
) -> None:
    row: dict[str, object] = dict(context)
    row.update(
        {
            "method": method,
            "objective_last": objective_last if objective_last is not None else math.nan,
            "rounds": rounds if rounds is not None else math.nan,
        }
    )
    row.update(metrics)
    rows.append(row)


def run_methods_once(
    clients: list[ClientData],
    n_clusters: int,
    seed: int,
    context: dict[str, object],
    rounds: int = 45,
    local_steps: int = 5,
    fuzzifier: float = 2.0,
    server_lr: float = 0.75,
    delta_lr: float = 0.7,
    delta_reg: float = 0.05,
    min_relevance: float = 0.05,
    fixed_relevance: float = 0.5,
    include_centralized: bool = True,
    include_local: bool = True,
    include_no_footprint: bool = True,
    include_model_averaging: bool = True,
    include_ifca: bool = True,
    ifca_groups: int = 3,
    include_fedprox: bool = True,
    include_ditto: bool = True,
    prox_coef: float = 0.1,
    ditto_coef: float = 0.1,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    x_all = np.vstack([client.x for client in clients])
    init = initialize_centers_from_data(x_all, n_clusters, random_state=seed + 77)
    artifacts: dict[str, object] = {"init_centers": init}

    if include_centralized:
        central = fcm(x_all, n_clusters, init_centers=init, m=fuzzifier, max_iter=200)
        add_result_row(
            rows,
            context,
            "Centralized FCM",
            evaluate_global_method(clients, central.centers, m=fuzzifier),
            central.objective_history[-1],
            central.n_iter,
        )
        artifacts["central_centers"] = central.centers

    if include_local:
        add_result_row(
            rows,
            context,
            "Local FCM",
            evaluate_local_fcm(clients, n_clusters, seed=seed, m=fuzzifier),
            None,
            None,
        )

    fed = federated_fcm(
        clients,
        init_centers=init,
        rounds=rounds,
        local_steps=local_steps,
        m=fuzzifier,
        server_lr=server_lr,
    )
    add_result_row(
        rows,
        context,
        "FedFCM",
        evaluate_global_method(clients, fed.centers, m=fuzzifier),
        fed.objective_history[-1],
        fed.n_rounds,
    )
    artifacts["fed_centers"] = fed.centers

    if include_model_averaging:
        model_avg = federated_fcm_model_averaging(
            clients,
            init_centers=init,
            m=fuzzifier,
            seed=seed + 51,
        )
        add_result_row(
            rows,
            context,
            "MA-FedFCM",
            evaluate_global_method(clients, model_avg.centers, m=fuzzifier),
            None,
            model_avg.n_rounds,
        )
        artifacts["model_avg_centers"] = model_avg.centers

    if include_ifca:
        init_models = [
            initialize_centers_from_data(x_all, n_clusters, random_state=seed + 211 + k)
            for k in range(ifca_groups)
        ]
        ifca_models, ifca_assignments = ifca_fcm(
            clients,
            init_models=init_models,
            rounds=rounds,
            local_steps=local_steps,
            m=fuzzifier,
            server_lr=server_lr,
        )
        ifca_centers = [ifca_models[ifca_assignments[p]] for p in range(len(clients))]
        add_result_row(
            rows,
            context,
            "IFCA-FCM",
            evaluate_personalized_method(clients, ifca_centers, m=fuzzifier),
            None,
            rounds,
        )
        artifacts["ifca_centers_by_client"] = ifca_centers

    if include_fedprox:
        prox = fedprox_fcm(
            clients,
            init_centers=init,
            rounds=rounds,
            local_steps=local_steps,
            m=fuzzifier,
            server_lr=server_lr,
            prox_coef=prox_coef,
        )
        add_result_row(
            rows,
            context,
            "FedProx-FCM",
            evaluate_global_method(clients, prox.centers, m=fuzzifier),
            None,
            prox.n_rounds,
        )
        artifacts["fedprox_centers"] = prox.centers

    if include_ditto:
        ditto_global, ditto_personal = ditto_fcm(
            clients,
            init_centers=init,
            rounds=rounds,
            local_steps=local_steps,
            m=fuzzifier,
            server_lr=server_lr,
            ditto_coef=ditto_coef,
        )
        add_result_row(
            rows,
            context,
            "Ditto-FCM",
            evaluate_personalized_method(clients, ditto_personal, m=fuzzifier),
            None,
            ditto_global.n_rounds,
        )
        artifacts["ditto_centers_by_client"] = ditto_personal

    if include_no_footprint:
        no_fp = personalized_federated_fcm(
            clients,
            init_centers=init,
            rounds=rounds,
            local_steps=local_steps,
            m=fuzzifier,
            server_lr=server_lr,
            delta_lr=delta_lr,
            delta_reg=delta_reg,
            use_footprint=False,
            fixed_relevance=fixed_relevance,
        )
        no_fp_centers = [no_fp.centers + no_fp.deltas[p] for p in range(len(clients))]
        no_fp_metrics = evaluate_personalized_method(clients, no_fp_centers, m=fuzzifier)
        no_fp_metrics.update(relevance_metrics(no_fp.relevance))
        add_result_row(
            rows,
            context,
            "PFedFCM(no-footprint)",
            no_fp_metrics,
            no_fp.objective_history[-1],
            no_fp.n_rounds,
        )
        artifacts["no_fp_centers_by_client"] = no_fp_centers

    gf = personalized_federated_fcm(
        clients,
        init_centers=init,
        rounds=rounds,
        local_steps=local_steps,
        m=fuzzifier,
        server_lr=server_lr,
        delta_lr=delta_lr,
        delta_reg=delta_reg,
        min_relevance=min_relevance,
        use_footprint=True,
    )
    gf_centers = [gf.centers + gf.deltas[p] for p in range(len(clients))]
    gf_metrics = evaluate_personalized_method(clients, gf_centers, m=fuzzifier)
    gf_metrics.update(relevance_metrics(gf.relevance))
    add_result_row(
        rows,
        context,
        "GF-PFedFCM",
        gf_metrics,
        gf.objective_history[-1],
        gf.n_rounds,
    )
    artifacts.update(
        {
            "gf_centers": gf.centers,
            "gf_centers_by_client": gf_centers,
            "gf_relevance": gf.relevance,
            "gf_footprint": gf.footprint,
        }
    )
    return rows, artifacts
