from __future__ import annotations

import csv
import math
from pathlib import Path

import enigma_sleep_cognition


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "example" / "synthetic_sleep_cognition.csv"
OUTPUT_DIR = ROOT / "results" / "example_outputs"
TARGET = "cognitive_score"
FEATURES = [
    "age",
    "sex_binary",
    "sleep_duration",
    "sleep_efficiency",
    "brain_feature_1",
    "brain_feature_2",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float_matrix(rows: list[dict[str, str]], features: list[str]) -> list[list[float]]:
    return [[float(row[feature]) for feature in features] for row in rows]


def as_float_vector(rows: list[dict[str, str]], target: str) -> list[float]:
    return [float(row[target]) for row in rows]


def train_test_split_rows(rows: list[dict[str, str]], n_test: int = 4) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return rows[:-n_test], rows[-n_test:]


def standardize_train_test(
    train_x: list[list[float]], test_x: list[list[float]]
) -> tuple[list[list[float]], list[list[float]]]:
    n_features = len(train_x[0])
    means = [sum(row[i] for row in train_x) / len(train_x) for i in range(n_features)]
    scales = []
    for i, mean in enumerate(means):
        variance = sum((row[i] - mean) ** 2 for row in train_x) / len(train_x)
        scales.append(math.sqrt(variance) or 1.0)

    def transform(matrix: list[list[float]]) -> list[list[float]]:
        return [[(row[i] - means[i]) / scales[i] for i in range(n_features)] for row in matrix]

    return transform(train_x), transform(test_x)


def fit_linear_model(x: list[list[float]], y: list[float], learning_rate: float = 0.03, steps: int = 2500) -> list[float]:
    weights = [0.0] * (len(x[0]) + 1)
    for _ in range(steps):
        gradients = [0.0] * len(weights)
        for row, target in zip(x, y):
            features = [1.0] + row
            error = sum(weight * value for weight, value in zip(weights, features)) - target
            for i, value in enumerate(features):
                gradients[i] += error * value / len(x)
        for i, gradient in enumerate(gradients):
            weights[i] -= learning_rate * gradient
    return weights


def predict(weights: list[float], x: list[list[float]]) -> list[float]:
    predictions = []
    for row in x:
        features = [1.0] + row
        predictions.append(sum(weight * value for weight, value in zip(weights, features)))
    return predictions


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((truth - pred) ** 2 for truth, pred in zip(y_true, y_pred))
    ss_tot = sum((truth - mean_y) ** 2 for truth in y_true)
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def mean_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    return sum(abs(truth - pred) for truth, pred in zip(y_true, y_pred)) / len(y_true)


def write_svg(y_true: list[float], y_pred: list[float], path: Path) -> None:
    width = 480
    height = 480
    pad = 64
    values = y_true + y_pred
    lo = min(values) - 1
    hi = max(values) + 1

    def scale_x(value: float) -> float:
        return pad + (value - lo) / (hi - lo) * (width - 2 * pad)

    def scale_y(value: float) -> float:
        return height - pad - (value - lo) / (hi - lo) * (height - 2 * pad)

    points = "\n".join(
        f'<circle cx="{scale_x(obs):.2f}" cy="{scale_y(pred):.2f}" r="6" fill="#2f6f73" />'
        for obs, pred in zip(y_true, y_pred)
    )
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#222" stroke-width="1.5"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#222" stroke-width="1.5"/>
  <line x1="{scale_x(lo):.2f}" y1="{scale_y(lo):.2f}" x2="{scale_x(hi):.2f}" y2="{scale_y(hi):.2f}" stroke="#999" stroke-width="1" stroke-dasharray="4 4"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="16">Smoke-test prediction check</text>
  <text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">Observed cognitive score</text>
  <text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" font-family="Arial, sans-serif" font-size="13">Predicted cognitive score</text>
  {points}
</svg>
''',
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(INPUT_CSV)
    train_rows, test_rows = train_test_split_rows(rows)
    train_x, test_x = standardize_train_test(
        as_float_matrix(train_rows, FEATURES),
        as_float_matrix(test_rows, FEATURES),
    )
    train_y = as_float_vector(train_rows, TARGET)
    test_y = as_float_vector(test_rows, TARGET)
    weights = fit_linear_model(train_x, train_y)
    predicted = predict(weights, test_x)

    metrics_path = OUTPUT_DIR / "smoke_test_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "package_version", "value": enigma_sleep_cognition.__version__})
        writer.writerow({"metric": "n_train", "value": len(train_rows)})
        writer.writerow({"metric": "n_test", "value": len(test_rows)})
        writer.writerow({"metric": "r2", "value": f"{r2_score(test_y, predicted):.6f}"})
        writer.writerow({"metric": "mae", "value": f"{mean_absolute_error(test_y, predicted):.6f}"})

    write_svg(test_y, predicted, OUTPUT_DIR / "smoke_test_prediction_plot.svg")

    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
