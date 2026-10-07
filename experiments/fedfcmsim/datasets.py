from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .synthetic import ClientData

OPENML_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "sklearn_data"

# Real UCI datasets fetched from OpenML (name -> version). These are distinct from
# the sklearn built-ins and provide genuine cross-domain feature spaces.
OPENML_FEATURE_DATASETS = {
    "pendigits": 1,   # pen-based handwritten digits, 16-d, 10 classes
    "optdigits": 1,   # optical handwritten digits, 64-d, 10 classes
    "satimage": 1,    # Statlog Landsat satellite, 36-d, 6 classes
    "letter": 1,      # letter recognition, 16-d, 26 classes, 20k samples
    "mnist_784": 1,   # full MNIST, 784-d, 10 classes, 70k samples
}


def _ensure_ssl_context() -> None:
    """Install a certifi-backed SSL context so OpenML downloads work on macOS.
    No-op once datasets are cached locally."""
    try:
        import ssl
        import urllib.request

        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
        urllib.request.install_opener(opener)
    except Exception:
        pass


def load_openml_feature(name: str) -> tuple[np.ndarray, np.ndarray, int]:
    version = OPENML_FEATURE_DATASETS[name]
    OPENML_CACHE.mkdir(parents=True, exist_ok=True)
    _ensure_ssl_context()
    data = fetch_openml(name, version=version, as_frame=False, data_home=str(OPENML_CACHE))
    x = StandardScaler().fit_transform(np.asarray(data.data, dtype=float))
    _, y = np.unique(np.asarray(data.target), return_inverse=True)
    return x, y.astype(int), int(len(np.unique(y)))


def dirichlet_partition(
    x: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    alpha: float,
    min_size: int = 5,
    random_state: int | None = None,
    max_attempts: int = 200,
) -> list[ClientData]:
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    labels = np.unique(y)

    for _ in range(max_attempts):
        client_indices = [[] for _ in range(n_clients)]
        for label in labels:
            idx = np.where(y == label)[0]
            rng.shuffle(idx)
            proportions = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(idx)).astype(int)
            for client_id, part in enumerate(np.split(idx, cuts)):
                client_indices[client_id].extend(part.tolist())

        sizes = [len(indices) for indices in client_indices]
        if min(sizes) >= min_size:
            clients = []
            for client_id, indices in enumerate(client_indices):
                indices = np.array(indices, dtype=int)
                rng.shuffle(indices)
                clients.append(ClientData(x=x[indices], y=y[indices], name=f"client_{client_id:02d}"))
            return clients

    raise RuntimeError(f"Could not create a Dirichlet partition with min_size={min_size}.")


def load_sklearn_dataset(name: str) -> tuple[np.ndarray, np.ndarray, int]:
    name = name.lower()
    if name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    elif name in {"breast_cancer", "breast-cancer", "cancer"}:
        data = load_breast_cancer()
    elif name == "digits":
        data = load_digits()
    else:
        raise ValueError(f"Unknown built-in dataset: {name}")

    x = StandardScaler().fit_transform(np.asarray(data.data, dtype=float))
    y = np.asarray(data.target)
    return x, y, len(np.unique(y))


def load_feature_dataset(name: str, random_state: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
    name = name.lower()
    if name.startswith("digits_pca"):
        suffix = name.removeprefix("digits_pca")
        n_components = int(suffix) if suffix else 16
        data = load_digits()
        x = StandardScaler().fit_transform(np.asarray(data.data, dtype=float))
        x = PCA(n_components=n_components, whiten=True, random_state=random_state).fit_transform(x)
        y = np.asarray(data.target)
        return x, y, len(np.unique(y))
    if name in {"digits_pixels", "digits_raw"}:
        data = load_digits()
        x = StandardScaler().fit_transform(np.asarray(data.data, dtype=float))
        y = np.asarray(data.target)
        return x, y, len(np.unique(y))
    if name.startswith("mnist784_pca"):
        # Full 70k-sample MNIST projected to a moderate PCA feature space.
        n_components = int(name.removeprefix("mnist784_pca") or 32)
        x, y, n_clusters = load_openml_feature("mnist_784")
        x = PCA(n_components=n_components, whiten=True, random_state=random_state or 0).fit_transform(x)
        return x, y, n_clusters
    if name in OPENML_FEATURE_DATASETS:
        return load_openml_feature(name)
    if name in {"wine_scaled", "wine"}:
        return load_sklearn_dataset("wine")
    if name in {"breast_cancer", "breast-cancer", "cancer"}:
        return load_sklearn_dataset("breast_cancer")
    if name == "iris":
        return load_sklearn_dataset("iris")
    raise ValueError(f"Unknown feature dataset: {name}")


def support_skew_partition(
    x: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    labels_per_client: int,
    random_state: int | None = None,
    min_size: int = 5,
) -> list[ClientData]:
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    labels = np.unique(y)
    n_labels = len(labels)
    labels_per_client = max(1, min(labels_per_client, n_labels))

    label_to_clients: dict[object, list[int]] = {label: [] for label in labels}
    for client_id in range(n_clients):
        for offset in range(labels_per_client):
            label = labels[(client_id + offset) % n_labels]
            label_to_clients[label].append(client_id)
    for label_idx, label in enumerate(labels):
        if not label_to_clients[label]:
            label_to_clients[label].append(label_idx % n_clients)

    client_indices = [[] for _ in range(n_clients)]
    for label in labels:
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        supported_clients = sorted(set(label_to_clients[label]))
        for client_id, part in zip(supported_clients, np.array_split(idx, len(supported_clients)), strict=True):
            client_indices[client_id].extend(part.tolist())

    clients = []
    sizes = [len(indices) for indices in client_indices]
    if min(sizes) < min_size:
        raise RuntimeError(
            f"Support-skew partition produced a client with fewer than {min_size} samples: {sizes}."
        )
    for client_id, indices in enumerate(client_indices):
        indices_array = np.array(indices, dtype=int)
        rng.shuffle(indices_array)
        clients.append(ClientData(x=x[indices_array], y=y[indices_array], name=f"client_{client_id:02d}"))
    return clients


def make_feature_clients(
    dataset: str,
    n_clients: int,
    partition: str,
    alpha: float = 0.1,
    labels_per_client: int = 3,
    random_state: int | None = None,
    min_size: int = 5,
) -> tuple[list[ClientData], int]:
    x, y, n_clusters = load_feature_dataset(dataset, random_state=random_state or 0)
    partition = partition.lower()
    if partition == "support_skew":
        clients = support_skew_partition(
            x=x,
            y=y,
            n_clients=n_clients,
            labels_per_client=labels_per_client,
            random_state=random_state,
            min_size=min_size,
        )
    elif partition.startswith("dirichlet"):
        if "_" in partition:
            alpha = float(partition.split("_", 1)[1])
        clients = dirichlet_partition(
            x=x,
            y=y,
            n_clients=n_clients,
            alpha=alpha,
            min_size=min_size,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown feature partition: {partition}")
    return clients, n_clusters


def make_tabular_clients(
    dataset: str,
    n_clients: int,
    alpha: float,
    random_state: int | None = None,
    min_size: int = 5,
) -> tuple[list[ClientData], int]:
    x, y, n_clusters = load_sklearn_dataset(dataset)
    clients = dirichlet_partition(
        x=x,
        y=y,
        n_clients=n_clients,
        alpha=alpha,
        min_size=min_size,
        random_state=random_state,
    )
    return clients, n_clusters
