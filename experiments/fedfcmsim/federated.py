from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fcm import (
    EPS,
    fcm,
    fcm_objective,
    initialize_centers_from_data,
    predict_membership,
    run_local_fcm_steps,
    squared_distances,
)
from .synthetic import ClientData


@dataclass
class FedFCMResult:
    centers: np.ndarray
    objective_history: list[float]
    n_rounds: int
    deltas: np.ndarray | None = None
    relevance: np.ndarray | None = None
    footprint: np.ndarray | None = None
    footprint_rho: np.ndarray | None = None


def _client_weights(clients: list[ClientData]) -> np.ndarray:
    counts = np.array([len(client.x) for client in clients], dtype=float)
    return counts / counts.sum()


def _total_objective(clients: list[ClientData], centers_by_client: list[np.ndarray], m: float) -> float:
    total = 0.0
    for client, centers in zip(clients, centers_by_client, strict=True):
        membership = predict_membership(client.x, centers, m=m)
        total += fcm_objective(client.x, centers, membership, m=m)
    return float(total)


def federated_fcm(
    clients: list[ClientData],
    init_centers: np.ndarray,
    rounds: int = 50,
    local_steps: int = 5,
    m: float = 2.0,
    server_lr: float = 0.8,
    tol: float = 1e-5,
    track_objective: bool = True,
) -> FedFCMResult:
    centers = np.asarray(init_centers, dtype=float).copy()
    weights = _client_weights(clients)
    history: list[float] = []

    for round_idx in range(rounds):
        previous = centers.copy()
        numerator = np.zeros_like(centers)
        denominator = np.zeros((centers.shape[0],), dtype=float)

        for weight, client in zip(weights, clients, strict=True):
            local = run_local_fcm_steps(client.x, centers, steps=local_steps, m=m)
            masses = np.sum(local.membership**m, axis=0)
            numerator += weight * masses[:, None] * local.centers
            denominator += weight * masses

        target = centers.copy()
        valid = denominator > EPS
        target[valid] = numerator[valid] / denominator[valid, None]
        centers = (1.0 - server_lr) * centers + server_lr * target
        if track_objective:
            history.append(_total_objective(clients, [centers] * len(clients), m=m))

        if np.linalg.norm(centers - previous) < tol:
            return FedFCMResult(centers=centers, objective_history=history, n_rounds=round_idx + 1)

    return FedFCMResult(centers=centers, objective_history=history, n_rounds=rounds)


def granular_footprint(
    x: np.ndarray,
    centers: np.ndarray,
    membership: np.ndarray,
    m: float = 2.0,
    grid_size: int = 60,
    radius_quantile: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.sqrt(np.maximum(squared_distances(x, centers), 0.0))
    footprint = np.zeros(centers.shape[0], dtype=float)
    rho = np.zeros(centers.shape[0], dtype=float)

    for j in range(centers.shape[0]):
        max_radius = float(np.quantile(distances[:, j], radius_quantile))
        max_radius = max(max_radius, EPS)
        radii = np.linspace(0.0, max_radius, grid_size)
        weights = membership[:, j] ** m

        best_value = -1.0
        best_rho = 0.0
        for radius in radii:
            coverage = float(np.sum(weights[distances[:, j] <= radius]) / len(x))
            specificity = 1.0 - float(radius / max_radius)
            value = coverage * specificity
            if value > best_value:
                best_value = value
                best_rho = float(radius / max_radius)

        footprint[j] = max(0.0, best_value)
        rho[j] = best_rho

    return footprint, rho


def _footprint_relevance(footprint: np.ndarray, min_relevance: float) -> np.ndarray:
    max_value = float(np.max(footprint))
    if max_value <= EPS:
        return np.full_like(footprint, min_relevance, dtype=float)
    return np.clip(footprint / max_value, min_relevance, 1.0)


def personalized_federated_fcm(
    clients: list[ClientData],
    init_centers: np.ndarray,
    rounds: int = 50,
    local_steps: int = 5,
    m: float = 2.0,
    server_lr: float = 0.7,
    delta_lr: float = 0.7,
    delta_reg: float = 0.05,
    min_relevance: float = 0.05,
    use_footprint: bool = True,
    fixed_relevance: float = 0.5,
    server_gate: bool = True,
    client_gate: bool = True,
    tol: float = 1e-5,
    footprint_grid_size: int = 60,
    track_objective: bool = True,
) -> FedFCMResult:
    centers = np.asarray(init_centers, dtype=float).copy()
    n_clients = len(clients)
    n_clusters, n_features = centers.shape
    deltas = np.zeros((n_clients, n_clusters, n_features), dtype=float)
    weights = _client_weights(clients)
    history: list[float] = []
    last_relevance = np.zeros((n_clients, n_clusters), dtype=float)
    last_footprint = np.zeros((n_clients, n_clusters), dtype=float)
    last_rho = np.zeros((n_clients, n_clusters), dtype=float)

    for round_idx in range(rounds):
        previous = centers.copy()
        numerator = np.zeros_like(centers)
        denominator = np.zeros((n_clusters,), dtype=float)

        for p, (weight, client) in enumerate(zip(weights, clients, strict=True)):
            personalized_centers = centers + deltas[p]
            local = run_local_fcm_steps(client.x, personalized_centers, steps=local_steps, m=m)

            if use_footprint:
                global_membership = predict_membership(client.x, centers, m=m)
                footprint, rho = granular_footprint(
                    client.x,
                    centers,
                    global_membership,
                    m=m,
                    grid_size=footprint_grid_size,
                )
                relevance = _footprint_relevance(footprint, min_relevance=min_relevance)
            else:
                footprint = np.ones(n_clusters, dtype=float) * fixed_relevance
                rho = np.zeros(n_clusters, dtype=float)
                relevance = np.ones(n_clusters, dtype=float) * fixed_relevance

            masses = np.sum(local.membership**m, axis=0)
            aggregation_relevance = relevance if server_gate else np.ones(n_clusters, dtype=float)
            aggregation_weight = weight * masses * aggregation_relevance
            numerator += aggregation_weight[:, None] * local.centers
            denominator += aggregation_weight

            desired_delta = local.centers - centers
            personalization_relevance = relevance if client_gate else np.ones(n_clusters, dtype=float)
            target_delta = personalization_relevance[:, None] * desired_delta
            deltas[p] = (1.0 - delta_lr) * deltas[p] + delta_lr * target_delta
            deltas[p] *= 1.0 - delta_reg

            last_relevance[p] = relevance
            last_footprint[p] = footprint
            last_rho[p] = rho

        target = centers.copy()
        valid = denominator > EPS
        target[valid] = numerator[valid] / denominator[valid, None]
        centers = (1.0 - server_lr) * centers + server_lr * target

        if track_objective:
            personalized_all = [centers + deltas[p] for p in range(n_clients)]
            history.append(_total_objective(clients, personalized_all, m=m))

        if np.linalg.norm(centers - previous) < tol:
            return FedFCMResult(
                centers=centers,
                objective_history=history,
                n_rounds=round_idx + 1,
                deltas=deltas,
                relevance=last_relevance,
                footprint=last_footprint,
                footprint_rho=last_rho,
            )

    return FedFCMResult(
        centers=centers,
        objective_history=history,
        n_rounds=rounds,
        deltas=deltas,
        relevance=last_relevance,
        footprint=last_footprint,
        footprint_rho=last_rho,
    )


def pedrycz_gradient_fcm(
    clients: list[ClientData],
    init_centers: np.ndarray,
    rounds: int = 50,
    local_steps: int = 10,
    m: float = 2.0,
    alpha: float = 0.05,
    tol: float = 1e-5,
    track_objective: bool = True,
) -> FedFCMResult:
    """Pedrycz's original gradient-based federated FCM (Pedrycz, "Federated FCM:
    Clustering Under Privacy Requirements", IEEE TFS 30(8), 2022, pp. 3384-3388).

    Per communication round (paper p.3385, steps 1-2):
      1. Server broadcasts global prototypes v_s.
      2. Each client ii sets vtilde_i = v_i and runs `local_steps` local FCM
         iterations alternating the membership update (Eq. 9) and the prototype
         update vtilde_i = sum_k u_ik^2 x_k / sum_k u_ik^2 (Eq. 10). The paper
         fixes the fuzzifier at m = 2, so memberships appear squared throughout;
         `local_steps` = 10 in the paper's experiments (p.3386-3387).
      3. The client computes the gradient of its local objective
         Q[ii] = sum_i sum_k u_ik[ii]^2 ||x_k[ii] - v_i||^2 (Eq. 2)
         with respect to each global prototype v_s (Eq. 7), using the
         POST-adaptation partition matrix U. Per feature j (Eq. 8):

             g_ii(v_sj) = -2 * sum_k u_ik^2 (x_kj[ii] - v_sj) / sigma_j^2[ii]

         where sigma_j^2[ii] is the variance of feature j on client ii
         (variance-normalized Euclidean distance, paper p.3385).
      4. Server applies the weighted gradient step (Eq. 5)

             v_s <- v_s - alpha * sum_ii beta_ii g_ii(v_s)

         with client weights beta_ii = N_ii / sum_jj N_jj given by SAMPLE
         COUNT (Eq. 6) -- not fuzzy mass.

    Since vtilde_i is the U-weighted mean, sum_k u_ik^2 (x_k - v_s) equals
    M_is * (vtilde_is - v_s) with M_is = sum_k u_ik^2, so the client update is
    implemented in that equivalent mean form.

    AMBIGUOUS: where in v-space Eq. (8) is evaluated. The paper says the client
    "computes the gradient of Q[ii] for the partition matrix U and vtilde_i";
    evaluated literally at v_s = vtilde_s the gradient is identically zero
    (vtilde is the U-weighted mean, Eq. 10), which would make the method a
    no-op. We therefore read g_ii(v_s) as the gradient FUNCTION formed from the
    post-adaptation U and evaluate it at the server's current prototypes
    v_s(iter) in Eq. 5 -- the only non-degenerate literal reading.
    AMBIGUOUS: near-zero sigma_j^2[ii] (a constant feature on a client) would
    blow up Eq. 8; we floor the per-client feature variance at EPS.
    AMBIGUOUS: the paper does not state whether sigma_j^2[ii] is recomputed per
    round; since each client's data are static, it is computed once and held
    fixed across rounds.
    AMBIGUOUS: the choice of alpha is not fully specified; it is exposed as a
    parameter and tuned only so that the IID sanity check matches centralized
    FCM.
    """
    centers = np.asarray(init_centers, dtype=float).copy()
    weights = _client_weights(clients)
    sigma2 = [np.maximum(np.var(client.x, axis=0), EPS) for client in clients]
    history: list[float] = []

    for round_idx in range(rounds):
        previous = centers.copy()
        gradient = np.zeros_like(centers)

        for weight, client, var in zip(weights, clients, sigma2, strict=True):
            local = run_local_fcm_steps(client.x, centers, steps=local_steps, m=m)
            masses = np.sum(local.membership**m, axis=0)
            gradient += weight * (-2.0) * masses[:, None] * (local.centers - centers) / var[None, :]

        centers = centers - alpha * gradient
        if track_objective:
            history.append(_total_objective(clients, [centers] * len(clients), m=m))

        if np.linalg.norm(centers - previous) < tol:
            return FedFCMResult(centers=centers, objective_history=history, n_rounds=round_idx + 1)

    return FedFCMResult(centers=centers, objective_history=history, n_rounds=rounds)


def _match_prototypes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a permutation mapping each source prototype to a target index by
    minimum-cost (Hungarian) assignment on Euclidean distance."""
    cost = np.sqrt(np.maximum(squared_distances(source, target), 0.0))
    perm = np.arange(source.shape[0])
    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)
        perm[row_ind] = col_ind
    except Exception:
        used: set[int] = set()
        for i in range(source.shape[0]):
            order = np.argsort(cost[i])
            for j in order:
                if int(j) not in used:
                    used.add(int(j))
                    perm[i] = int(j)
                    break
    return perm


def federated_fcm_model_averaging(
    clients: list[ClientData],
    init_centers: np.ndarray,
    rounds: int = 3,
    m: float = 2.0,
    local_max_iter: int = 80,
    server_lr: float = 1.0,
    seed: int = 0,
    tol: float = 1e-5,
) -> FedFCMResult:
    """Model-averaging federated fuzzy clustering (Wang et al., 2023, model-averaging
    variant). Each client fits a local FCM to convergence on its own data; the server
    aligns local prototypes to the current global set by minimum-cost matching and
    averages them with fuzzy-mass weights. This is the federated baseline known to
    degrade under non-IID/support-skew because independent local optima need not agree."""
    centers = np.asarray(init_centers, dtype=float).copy()
    weights = _client_weights(clients)
    n_clusters = centers.shape[0]
    rng = np.random.default_rng(seed)

    local_models: list[tuple[np.ndarray, np.ndarray]] = []
    for client in clients:
        local_init = initialize_centers_from_data(
            client.x, n_clusters, random_state=int(rng.integers(0, 2**31 - 1))
        )
        result = fcm(client.x, n_clusters, init_centers=local_init, m=m, max_iter=local_max_iter)
        masses = np.sum(result.membership**m, axis=0)
        local_models.append((result.centers, masses))

    for _ in range(rounds):
        numerator = np.zeros_like(centers)
        denominator = np.zeros((n_clusters,), dtype=float)
        for weight, (local_centers, local_masses) in zip(weights, local_models, strict=True):
            perm = _match_prototypes(local_centers, centers)
            aligned_centers = np.zeros_like(centers)
            aligned_masses = np.zeros((n_clusters,), dtype=float)
            for local_idx, global_idx in enumerate(perm):
                aligned_centers[global_idx] = local_centers[local_idx]
                aligned_masses[global_idx] = local_masses[local_idx]
            numerator += weight * aligned_masses[:, None] * aligned_centers
            denominator += weight * aligned_masses
        target = centers.copy()
        valid = denominator > EPS
        target[valid] = numerator[valid] / denominator[valid, None]
        previous = centers.copy()
        centers = (1.0 - server_lr) * centers + server_lr * target
        if np.linalg.norm(centers - previous) < tol:
            break

    return FedFCMResult(centers=centers, objective_history=[], n_rounds=rounds)


def fedprox_fcm(
    clients: list[ClientData],
    init_centers: np.ndarray,
    rounds: int = 45,
    local_steps: int = 5,
    m: float = 2.0,
    server_lr: float = 0.75,
    prox_coef: float = 0.1,
    tol: float = 1e-5,
) -> FedFCMResult:
    """FedProx-style federated FCM (Li et al., 2020, adapted to fuzzy prototypes).
    Local prototype updates carry a proximal pull toward the broadcast global
    prototypes: c_j = (sum u^m x + mu v_j) / (sum u^m + mu), with mu scaled by the
    client sample count. The server aggregates fuzzy-mass-weighted local estimates
    as in F-FCM. This is the canonical proximal-regularization PFL baseline."""
    centers = np.asarray(init_centers, dtype=float).copy()
    weights = _client_weights(clients)

    for round_idx in range(rounds):
        previous = centers.copy()
        numerator = np.zeros_like(centers)
        denominator = np.zeros((centers.shape[0],), dtype=float)

        for weight, client in zip(weights, clients, strict=True):
            mu = prox_coef * len(client.x)
            local_centers = centers.copy()
            for _ in range(local_steps):
                membership = predict_membership(client.x, local_centers, m=m)
                um = membership**m
                masses = np.sum(um, axis=0)
                for j in range(centers.shape[0]):
                    denom = masses[j] + mu
                    local_centers[j] = (um[:, j] @ client.x + mu * centers[j]) / denom
            membership = predict_membership(client.x, local_centers, m=m)
            masses = np.sum(membership**m, axis=0)
            numerator += weight * masses[:, None] * local_centers
            denominator += weight * masses

        target = centers.copy()
        valid = denominator > EPS
        target[valid] = numerator[valid] / denominator[valid, None]
        centers = (1.0 - server_lr) * centers + server_lr * target
        if np.linalg.norm(centers - previous) < tol:
            return FedFCMResult(centers=centers, objective_history=[], n_rounds=round_idx + 1)

    return FedFCMResult(centers=centers, objective_history=[], n_rounds=rounds)


def ditto_fcm(
    clients: list[ClientData],
    init_centers: np.ndarray,
    rounds: int = 45,
    local_steps: int = 5,
    m: float = 2.0,
    server_lr: float = 0.75,
    ditto_coef: float = 0.1,
    tol: float = 1e-5,
) -> tuple[FedFCMResult, list[np.ndarray]]:
    """Ditto-style personalized federated FCM (Li et al., 2021, adapted to fuzzy
    prototypes). A global prototype set is trained exactly as in F-FCM; in parallel,
    each client maintains personal prototypes updated by proximal-regularized FCM
    steps that pull toward the current global prototypes. Clients are evaluated
    with their personal prototypes. This is the canonical two-track proximal
    personalization baseline."""
    centers = np.asarray(init_centers, dtype=float).copy()
    weights = _client_weights(clients)
    personal = [np.asarray(init_centers, dtype=float).copy() for _ in clients]

    for round_idx in range(rounds):
        previous = centers.copy()
        numerator = np.zeros_like(centers)
        denominator = np.zeros((centers.shape[0],), dtype=float)

        for p, (weight, client) in enumerate(zip(weights, clients, strict=True)):
            # Global track: plain local FCM steps from the broadcast prototypes.
            local = run_local_fcm_steps(client.x, centers, steps=local_steps, m=m)
            masses = np.sum(local.membership**m, axis=0)
            numerator += weight * masses[:, None] * local.centers
            denominator += weight * masses

            # Personal track: proximal FCM steps pulled toward the global prototypes.
            lam = ditto_coef * len(client.x)
            w = personal[p]
            for _ in range(local_steps):
                membership = predict_membership(client.x, w, m=m)
                um = membership**m
                w_masses = np.sum(um, axis=0)
                for j in range(centers.shape[0]):
                    denom = w_masses[j] + lam
                    w[j] = (um[:, j] @ client.x + lam * centers[j]) / denom
            personal[p] = w

        target = centers.copy()
        valid = denominator > EPS
        target[valid] = numerator[valid] / denominator[valid, None]
        centers = (1.0 - server_lr) * centers + server_lr * target
        if np.linalg.norm(centers - previous) < tol:
            return (
                FedFCMResult(centers=centers, objective_history=[], n_rounds=round_idx + 1),
                personal,
            )

    return FedFCMResult(centers=centers, objective_history=[], n_rounds=rounds), personal


def ifca_fcm(
    clients: list[ClientData],
    init_models: list[np.ndarray],
    rounds: int = 45,
    local_steps: int = 5,
    m: float = 2.0,
    server_lr: float = 0.75,
    tol: float = 1e-5,
) -> tuple[list[np.ndarray], list[int]]:
    """Iterative federated clustering (IFCA, Ghosh et al., 2020) instantiated with
    fuzzy c-means local solvers. The server maintains K prototype sets; each client
    selects the set with the lowest local FCM objective and updates only that set.
    Each client is then evaluated with its selected set, giving a personalized
    (cluster-of-clients) baseline that does not use granular footprint relevance."""
    n_groups = len(init_models)
    models = [np.asarray(v, dtype=float).copy() for v in init_models]
    weights = _client_weights(clients)
    assignments = [0] * len(clients)

    def _best_model(client: ClientData) -> int:
        best_k = 0
        best_obj = np.inf
        for k in range(n_groups):
            membership = predict_membership(client.x, models[k], m=m)
            obj = fcm_objective(client.x, models[k], membership, m=m)
            if obj < best_obj:
                best_obj = obj
                best_k = k
        return best_k

    for _ in range(rounds):
        numerator = [np.zeros_like(models[0]) for _ in range(n_groups)]
        denominator = [np.zeros((models[0].shape[0],), dtype=float) for _ in range(n_groups)]
        for p, (weight, client) in enumerate(zip(weights, clients, strict=True)):
            best_k = _best_model(client)
            assignments[p] = best_k
            local = run_local_fcm_steps(client.x, models[best_k], steps=local_steps, m=m)
            masses = np.sum(local.membership**m, axis=0)
            numerator[best_k] += weight * masses[:, None] * local.centers
            denominator[best_k] += weight * masses

        moved = 0.0
        for k in range(n_groups):
            target = models[k].copy()
            valid = denominator[k] > EPS
            target[valid] = numerator[k][valid] / denominator[k][valid, None]
            updated = (1.0 - server_lr) * models[k] + server_lr * target
            moved = max(moved, float(np.linalg.norm(updated - models[k])))
            models[k] = updated
        if moved < tol:
            break

    for p, client in enumerate(clients):
        assignments[p] = _best_model(client)
    return models, assignments


def stallmann_avg2_fcm(
    clients: list[ClientData],
    init_centers: np.ndarray,
    rounds: int = 50,
    local_steps: int = 5,
    m: float = 2.0,
    seed: int = 0,
    kmeans_n_init: int = 10,
    tol: float = 1e-5,
    track_objective: bool = True,
) -> FedFCMResult:
    """Stallmann & Wilbik FFCM "avg2" aggregation ("Towards Federated Clustering:
    A Federated Fuzzy c-Means Algorithm (FFCM)", arXiv:2201.07316, Eq. 9).

    Each round, every client runs `local_steps` local FCM iterations starting
    from the broadcast prototypes V and reports its c resulting local centres.
    The server pools all P*c local centres, runs k-means with k = c over them,
    and the k-means centroids become the new global prototypes. The Eq. (7)
    client weights are NOT used by avg2 (they belong to the paper's weighted
    variants only).

    AMBIGUOUS: the paper does not specify the k-means initialisation. We use
    k-means++ seeded deterministically per round: rng = default_rng(seed) and
    each round consumes one integer draw passed to KMeans(random_state=...).
    AMBIGUOUS: n_init is unspecified; we use kmeans_n_init=10 restarts and keep
    the best-inertia run (the pooled point set is only P*c points, so this is
    cheap).
    AMBIGUOUS: empty k-means clusters are not discussed; we rely on sklearn's
    default behaviour (reassign the farthest point to the empty cluster).
    AMBIGUOUS: the paper replaces the prototypes outright; we do the same (no
    server learning rate / damping).
    """
    from sklearn.cluster import KMeans

    centers = np.asarray(init_centers, dtype=float).copy()
    n_clusters = centers.shape[0]
    rng = np.random.default_rng(seed)
    history: list[float] = []

    for round_idx in range(rounds):
        previous = centers.copy()
        pooled = []
        for client in clients:
            local = run_local_fcm_steps(client.x, centers, steps=local_steps, m=m)
            pooled.append(local.centers)
        pooled_centers = np.vstack(pooled)

        km = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=kmeans_n_init,
            max_iter=300,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        km.fit(pooled_centers)
        centers = np.asarray(km.cluster_centers_, dtype=float)

        if track_objective:
            history.append(_total_objective(clients, [centers] * len(clients), m=m))

        if np.linalg.norm(centers - previous) < tol:
            return FedFCMResult(centers=centers, objective_history=history, n_rounds=round_idx + 1)

    return FedFCMResult(centers=centers, objective_history=history, n_rounds=rounds)


def fednova_fcm(
    clients: list[ClientData],
    init_centers: np.ndarray,
    rounds: int = 50,
    local_steps: int = 5,
    m: float = 2.0,
    client_steps: list[int] | None = None,
    server_lr: float = 1.0,
    tol: float = 1e-5,
    track_objective: bool = True,
) -> FedFCMResult:
    """FedNova-style normalisation (Wang et al., NeurIPS 2020, "Tackling the
    Objective Inconsistency Problem in Heterogeneous Federated Optimization")
    adapted to prototype updates.

    Client p runs tau_p local FCM steps from the broadcast V and reports the
    normalised progress d_p = (V - V_p^end) / tau_p. The server aggregates

        V <- V - server_lr * tau_eff * sum_p beta_p * d_p,
        tau_eff = sum_p beta_p * tau_p,

    with beta_p the sample-count weights. With equal tau_p and server_lr=1 this
    reduces exactly to FedAvg on the local centres: V <- sum_p beta_p V_p^end.

    `client_steps` optionally supplies a per-client list of local step counts
    tau_p (the heterogeneous-local-step case FedNova is designed for);
    `local_steps` is used for every client when it is None.

    AMBIGUOUS: FedNova's d_p is a normalised SGD gradient sum; here it is the
    normalised DISPLACEMENT of the FCM alternating-minimisation iteration.
    FCM's centre update is closed-form, not a gradient step, so FedNova's
    objective-consistency guarantee does not transfer -- this is the natural
    prototype-space analogue only.
    AMBIGUOUS: beta_p follows the repo's FedAvg convention (N_p / sum N); the
    FedNova paper uses the same weighting for its aggregation.
    AMBIGUOUS: server_lr is an extra damping factor absent from the original
    formulation; server_lr=1.0 gives the literal update above. It is exposed
    for calibration because the displacement analogue need not share SGD's
    stable step-size range.
    """
    centers = np.asarray(init_centers, dtype=float).copy()
    weights = _client_weights(clients)
    if client_steps is None:
        taus = np.full(len(clients), float(local_steps))
    else:
        if len(client_steps) != len(clients):
            raise ValueError("client_steps must have one entry per client.")
        taus = np.asarray(client_steps, dtype=float)
    tau_eff = float(weights @ taus)
    history: list[float] = []

    for round_idx in range(rounds):
        previous = centers.copy()
        agg = np.zeros_like(centers)

        for weight, client, tau_p in zip(weights, clients, taus, strict=True):
            local = run_local_fcm_steps(client.x, centers, steps=int(tau_p), m=m)
            agg += weight * (centers - local.centers) / tau_p

        centers = centers - server_lr * tau_eff * agg
        if track_objective:
            history.append(_total_objective(clients, [centers] * len(clients), m=m))

        if np.linalg.norm(centers - previous) < tol:
            return FedFCMResult(centers=centers, objective_history=history, n_rounds=round_idx + 1)

    return FedFCMResult(centers=centers, objective_history=history, n_rounds=rounds)


def scffcm(
    clients: list[ClientData],
    init_centers: np.ndarray,
    rounds: int = 50,
    local_steps: int = 5,
    m: float = 2.0,
    eta_l: float = 0.2,
    eta_g: float = 2.0,
    cfraction: float = 1.0,
    seed: int = 0,
    tol: float = 1e-5,
    track_objective: bool = True,
) -> FedFCMResult:
    """SC-FFCM (Zhang, Deng, Zhang, Zhao, Xiao, Choi et al., "Robust Federated
    Fuzzy C-Means Algorithm in Heterogeneous Scenarios", IEEE TFS 33(9):
    3168-3181, 2025, DOI 10.1109/TFUZZ.2025.3584697), implemented from the
    authors' public code (github.com/Creazy-MR/SC-FFCM). The paper PDF was not
    obtained; choices below that rest on code rather than the paper are marked
    AMBIGUOUS.

    Structure (code-verbatim): gradient FCM on each client -- NOT the closed-form
    centroid. Per local step k = 1..K the client recomputes the membership at
    its current prototypes and takes a control-variate-corrected gradient step

        U   <- membership(V_p)                         (standard Bezdek Eq.)
        g_k <- (2/N_p) * sum_s u_sk^m * (v_k - x_s)    (gradient of (1/N_p) J)
        V_p <- V_p - eta_l * (g - c_p + c)

    with client variate c_p and server variate c FROZEN during the K steps.
    After K steps (SCAFFOLD Option II -- the code has no Option I):

        c_p^+    = c_p - c + (V_start - V_end) / (K * eta_l)
        Delta_c_p = c_p^+ - c_p ;  c_p <- c_p^+
        Delta_V_p = V_end - V_start

    Server (equal weights 1/P -- sample counts are NOT used):

        V <- V + eta_g * (1/P) * sum_{p in S_t} Delta_V_p
        c <- c + cfraction * (1/P) * sum_{p in S_t} Delta_c_p

    c and every c_p are initialised to zero.

    Deviations from the authors' code (required by our protocol):
      * zero-distance guard in the membership update via predict_membership
        (the original has none and can divide by zero);
      * the round-0 co_init bug is NOT reproduced: the original first round
        measures Delta_V against uninitialised random server prototypes; we
        initialise every client from the given init_centers like every other
        method in this package;
      * cfraction defaults to 1.0 (full participation, matching our other
        baselines) instead of the code's argparse default 0.5; when
        cfraction < 1, P = int(M * cfraction) clients are drawn per round via
        rng = default_rng(seed);
      * the code's one-hot membership init is immaterial (U is recomputed at
        the current V before the first gradient step) and is not reproduced.

    AMBIGUOUS: eta_l / eta_g defaults. The code's argparse values
    (eta_l=0.2, eta_g=0.5, K=50) were tuned on MinMax-scaled iris; our data are
    StandardScaler'd, so the defaults here were recalibrated on the `iid`
    synthetic scenario only (scripts/T6_calibrate.py, 2 seeds, K=local_steps=5,
    grid eta_l in {0.02,0.05,0.1,0.2} x eta_g in {0.25,0.5,1.0,2.0}). Frozen at
    eta_l=0.2, eta_g=2.0: the only grid cell reaching centralized-FCM accuracy
    AND objective on both seeds; every eta_g <= 0.5 setting diverged.
    AMBIGUOUS: K (paper/code default 50) is mapped onto `local_steps`
    (protocol default 5) so the per-round local budget matches the other
    baselines.
    AMBIGUOUS: the paper's numbered equations were not obtained; the gradient
    sign, the (1/N_i) normalisation and the Option-II variate update above are
    taken from the code, not from the paper text.
    """
    centers = np.asarray(init_centers, dtype=float).copy()
    n_clients = len(clients)
    c_global = np.zeros_like(centers)
    c_local = [np.zeros_like(centers) for _ in clients]
    rng = np.random.default_rng(seed)
    history: list[float] = []
    n_part = max(1, int(n_clients * cfraction))

    for round_idx in range(rounds):
        previous = centers.copy()
        if n_part >= n_clients:
            selected = list(range(n_clients))
        else:
            selected = sorted(rng.choice(n_clients, size=n_part, replace=False).tolist())

        delta_v_sum = np.zeros_like(centers)
        delta_c_sum = np.zeros_like(centers)
        for p in selected:
            client = clients[p]
            n_i = float(len(client.x))
            v_start = centers.copy()
            v = centers.copy()
            for _ in range(local_steps):
                membership = predict_membership(client.x, v, m=m)
                um = membership**m
                masses = np.sum(um, axis=0)
                gradient = (2.0 / n_i) * (masses[:, None] * v - um.T @ client.x)
                v = v - eta_l * (gradient - c_local[p] + c_global)

            c_new = c_local[p] - c_global + (v_start - v) / (local_steps * eta_l)
            delta_c_sum += c_new - c_local[p]
            c_local[p] = c_new
            delta_v_sum += v - v_start

        centers = centers + eta_g * delta_v_sum / n_part
        c_global = c_global + cfraction * delta_c_sum / n_part

        if track_objective:
            history.append(_total_objective(clients, [centers] * len(clients), m=m))

        if np.linalg.norm(centers - previous) < tol:
            return FedFCMResult(centers=centers, objective_history=history, n_rounds=round_idx + 1)

    return FedFCMResult(centers=centers, objective_history=history, n_rounds=rounds)
