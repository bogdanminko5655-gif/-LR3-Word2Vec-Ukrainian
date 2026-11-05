"""Офлайн-тести на синтетичній моделі — це НЕ accuracy реального Word2Vec."""
import numpy as np
import pandas as pd
import pytest
from gensim.models import KeyedVectors

from lab3 import (
    cosine_similarity, euclidean_distance, find_analogy,
    get_country, get_accuracy, load_capital_analogies,
)


def synthetic_model():
    model = KeyedVectors(vector_size=3)
    words = ["Греція", "Афіни", "Україна", "Київ", "Польща", "Варшава", "слово"]
    vectors = np.array([
        [1, 0, 1], [1, 0, 0],
        [0, 1, 1], [0, 1, 0],
        [0, 2, 1], [0, 2, 0],
        [-1, -1, -1],
    ], dtype=np.float32)
    model.add_vectors(words, vectors)
    return model


def test_math():
    assert cosine_similarity(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.0)
    assert cosine_similarity(np.array([1, 2]), np.array([1, 2])) == pytest.approx(1.0)
    assert euclidean_distance(np.array([1, 2]), np.array([4, 6])) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        cosine_similarity(np.zeros(2), np.ones(2))


def test_analogy_exact():
    model = synthetic_model()
    answer = get_country("Афіни", "Греція", "Київ", model)
    assert answer is not None and answer[0] == "Україна"
    assert find_analogy("Київ", "Україна", "Афіни", model) == "Греція"
    assert get_country("Невідоме_місто", "Греція", "Київ", model) is None


def test_accuracy_counts_missing_as_wrong():
    model = synthetic_model()
    data = pd.DataFrame([
        ("Греція", "Афіни", "Україна", "Київ"),
        ("Україна", "Київ", "Греція", "Афіни"),
        ("Греція", "Афіни", "Невідома", "Київ"),
    ], columns=["country1", "city1", "country2", "city2"])
    metrics, predictions = get_accuracy(model, data, batch_size=2, vocab_block=3)
    assert metrics["correct"] == 2
    assert metrics["total"] == 3
    assert metrics["evaluated"] == 2
    assert metrics["accuracy_all"] == pytest.approx(2 / 3)
    assert metrics["accuracy_covered"] == pytest.approx(1.0)
    assert predictions.iloc[2]["status"] == "missing_target"


def test_dataset_only_capital_section(tmp_path):
    path = tmp_path / "test_vocabulary.txt"
    path.write_text(": country-capital\nУкраїна\tКиїв\tГреція\tАфіни\n"
                    ": country-region\nУкраїна\tЄвропа\tГреція\tЄвропа\n", encoding="utf-8")
    rows = load_capital_analogies(path)
    assert len(rows) == 1
    assert rows.iloc[0]["city2"] == "Афіни"
