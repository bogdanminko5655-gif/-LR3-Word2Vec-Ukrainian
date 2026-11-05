"""ЛР №3. Українські вектори слів Word2Vec, аналогії та PCA.

Приклад запуску:
    python lab3.py --source original
    python lab3.py --source huggingface --max-tests 100

Дані/модель завантажуються з оригінальних ресурсів авторів, не входять до архіву.
"""
from __future__ import annotations

import argparse
import bz2
import json
import math
import shutil
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Збереження графіків без графічного інтерфейсу.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent
DATA_URL = "https://raw.githubusercontent.com/lang-uk/vecs/refs/heads/master/test/test_vocabulary.txt"
MODEL_URL = "https://lang.org.ua/static/downloads/models/ubercorpus.cased.tokenized.word2vec.300d.bz2"
MODEL_NAME = "ubercorpus.cased.tokenized.word2vec.300d"
DATA_FILE = BASE_DIR / "data" / "test_vocabulary.txt"
MODEL_FILE = BASE_DIR / "models" / MODEL_NAME
SELECTED_WORDS = ["Україна", "Київ", "Польща", "Варшава", "Німеччина", "Берлін", "Франція", "Париж"]


def download_file(url: str, destination: Path) -> Path:
    """Завантажує файл потоково та прибирає неповний файл у разі помилки."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    temporary = destination.with_name(destination.name + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "LR3-Word2Vec/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def download_dataset(destination: Path = DATA_FILE) -> Path:
    return download_file(DATA_URL, destination)


def load_capital_analogies(path: Path) -> pd.DataFrame:
    """Читає ЛИШЕ перший розділ ': country-capital' (4271 рядок)."""
    records: list[list[str]] = []
    section = None
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(":"):
                label = line[1:].strip()
                if section == "country-capital":
                    break
                section = label
                continue
            if section != "country-capital":
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(f"Очікувалося 4 поля у рядку: {line[:150]}")
            records.append(parts)
    if not records:
        raise ValueError("У файлі немає розділу ': country-capital' або він порожній")
    return pd.DataFrame(records, columns=["country1", "city1", "country2", "city2"])


def obtain_model(source: str = "original", destination: Path = MODEL_FILE) -> Path:
    """Готує текстовий файл моделі. Альтернатива: HF lang-uk/word2vec-uk."""
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == "original":
        archive = destination.with_suffix(destination.suffix + ".bz2")
        try:
            print("Завантаження архіву Word2Vec (оригінальне посилання)...", flush=True)
            download_file(MODEL_URL, archive)
            print("Розпакування bz2: потрібен додатковий простір на диску...", flush=True)
            temporary = destination.with_name(destination.name + ".part")
            try:
                with bz2.open(archive, "rb") as compressed, temporary.open("wb") as out:
                    shutil.copyfileobj(compressed, out, length=1024 * 1024)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            archive.unlink(missing_ok=True)  # Економія місця після розпакування.
            return destination
        except Exception as exc:
            print(f"Оригінальне джерело недоступне ({exc}). Пробую Hugging Face...", flush=True)
    if source not in {"original", "huggingface"}:
        raise ValueError("source має бути 'original' або 'huggingface'")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id="lang-uk/word2vec-uk", filename="ubercorpus.cased.tokenized.300d"))


def load_model(path: Path) -> KeyedVectors:
    print("Завантаження Word2Vec у пам'ять (потрібно декілька ГБ RAM)...", flush=True)
    model = KeyedVectors.load_word2vec_format(str(path), binary=False, unicode_errors="strict")
    print(f"Слів у словнику: {len(model):,}; розмірність: {model.vector_size}", flush=True)
    return model


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """cos(u,v) = dot(u,v) / (||u|| * ||v||)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError("Вектори мають різну розмірність")
    denominator = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denominator == 0:
        raise ValueError("Косинусна подібність не визначена для нульового вектора")
    return float(np.dot(u, v) / denominator)


def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    """d(u,v) = sqrt(sum((u_i-v_i)^2))."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError("Вектори мають різну розмірність")
    return float(np.linalg.norm(u - v))


def get_country(city1: str, country1: str, city2: str, model: KeyedVectors) -> tuple[str, float] | None:
    """Шукає країну 2 за правилом країна1 - столиця1 + столиця2.

    Пошук здійснюється по ВСЬОМУ словнику, а не лише серед країн.
    Три слова з умови виключаються з можливих відповідей.
    """
    excluded = {city1, country1, city2}
    if not all(word in model for word in excluded):
        return None
    target = model[country1] - model[city1] + model[city2]
    if not np.linalg.norm(target):
        return None
    # Щонайменше 3 найкращі відповіді можуть бути початковими словами.
    topn = min(len(model), len(excluded) + 1)
    for word, similarity in model.similar_by_vector(target, topn=topn):
        if word not in excluded:
            return word, float(similarity)
    return None


def find_analogy(w1: str, w2: str, w3: str, model: KeyedVectors) -> str | None:
    """w1 = столиця 1, w2 = країна 1, w3 = столиця 2."""
    answer = get_country(w1, w2, w3, model)
    return answer[0] if answer is not None else None


def get_accuracy(
    model: KeyedVectors,
    data: pd.DataFrame,
    batch_size: int = 48,
    vocab_block: int = 40_000,
) -> tuple[dict, pd.DataFrame]:
    """Точно шукає найближчі за косинусом слова у повному словнику.

    Для швидкодії використовує блокове матричне множення BLAS замість
    повільного Python-циклу по всіх словах для кожного прикладу.
    Навіть так повний запуск 4271 тестів на великій моделі може бути тривалим.
    """
    if batch_size < 1 or vocab_block < 1:
        raise ValueError("batch_size і vocab_block мають бути додатними")
    results: list[dict] = []
    valid_indices: list[int] = []
    queries: list[np.ndarray] = []
    source_indices: list[tuple[int, ...]] = []
    total = len(data)

    for i, row in enumerate(data.itertuples(index=False)):
        country1, city1, country2, city2 = row
        entry = {"country1": country1, "city1": city1,
                 "country2": country2, "city2": city2,
                 "predicted_country2": "", "similarity": None,
                 "correct": False, "status": "ok"}
        results.append(entry)
        source_words = (city1, country1, city2)
        missing_inputs = [w for w in source_words if w not in model]
        if missing_inputs:
            entry["status"] = "missing_input"
            continue
        if country2 not in model:
            entry["status"] = "missing_target"
            continue
        vec = np.asarray(model[country1] - model[city1] + model[city2], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if not math.isfinite(norm) or norm == 0:
            entry["status"] = "zero_query"
            continue
        valid_indices.append(i)
        queries.append(vec / norm)
        source_indices.append(tuple(model.key_to_index[w] for w in set(source_words)))

    evaluated = len(queries)
    print(f"Усього прикладів: {total}; доступні всі 4 слова: {evaluated}", flush=True)
    if queries:
        print("Підготовка нормованих векторів словника...", flush=True)
        normalized_vocab = model.get_normed_vectors()
        vocab_count = len(model)
        for q_start in range(0, evaluated, batch_size):
            q_stop = min(q_start + batch_size, evaluated)
            q_batch = np.ascontiguousarray(np.array(queries[q_start:q_stop], dtype=np.float32).T)
            current_n = q_stop - q_start
            best_score = np.full(current_n, -np.inf, dtype=np.float32)
            best_idx = np.full(current_n, -1, dtype=np.int64)
            for v_start in range(0, vocab_count, vocab_block):
                v_stop = min(v_start + vocab_block, vocab_count)
                sim = normalized_vocab[v_start:v_stop] @ q_batch
                for j in range(current_n):
                    for blocked_index in source_indices[q_start + j]:
                        if v_start <= blocked_index < v_stop:
                            sim[blocked_index - v_start, j] = -np.inf
                col = np.arange(current_n)
                arg = np.argmax(sim, axis=0)
                values = sim[arg, col]
                improved = values > best_score
                best_score[improved] = values[improved]
                best_idx[improved] = v_start + arg[improved]
            for j in range(current_n):
                row_idx = valid_indices[q_start + j]
                if best_idx[j] >= 0:
                    predicted = model.index_to_key[int(best_idx[j])]
                    results[row_idx]["predicted_country2"] = predicted
                    results[row_idx]["similarity"] = float(best_score[j])
                    results[row_idx]["correct"] = (predicted == results[row_idx]["country2"])
            print(f"  Перевірено {q_stop}/{evaluated} доступних прикладів", flush=True)

    result_df = pd.DataFrame(results)
    correct = int(result_df["correct"].sum())
    metrics = {
        "total": total,
        "evaluated": evaluated,
        "correct": correct,
        "accuracy_all": float(correct / total) if total else 0.0,
        "accuracy_covered": float(correct / evaluated) if evaluated else 0.0,
        "coverage": float(evaluated / total) if total else 0.0,
        "missing_input": sum(r["status"] == "missing_input" for r in results),
        "missing_target": sum(r["status"] == "missing_target" for r in results),
        "zero_query": sum(r["status"] == "zero_query" for r in results),
    }
    return metrics, result_df


def visualize(words: list[str], model: KeyedVectors, filename: Path) -> list[dict]:
    """Зберігає PNG і повертає координати PCA для вебсторінки."""
    valid_words = list(dict.fromkeys(w for w in words if w in model))
    if len(valid_words) < 3:
        print(f"PCA пропущено: доступно лише {len(valid_words)} слів.")
        return []
    word_vectors = np.array([model[w] for w in valid_words], dtype=np.float32)
    pca = PCA(n_components=2)
    points = pca.fit_transform(word_vectors)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.scatter(points[:, 0], points[:, 1], s=45, alpha=0.78)
    for (x, y), word in zip(points, valid_words):
        ax.annotate(word, (float(x), float(y)), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.axhline(0, linewidth=0.5, alpha=0.3)
    ax.axvline(0, linewidth=0.5, alpha=0.3)
    ax.set_title("Українські слова у просторі PCA (2D)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсії)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсії)")
    ax.grid(alpha=0.2)
    fig.savefig(filename, dpi=170)
    plt.close(fig)
    return [{"word": word, "x": float(x), "y": float(y)}
            for word, (x, y) in zip(valid_words, points)]


def save_results(metrics: dict, predictions: pd.DataFrame,
                 selected_points: list[dict], all_points: list[dict],
                 examples: list[dict], output_dir: Path = BASE_DIR / "results") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    report = f"""# Звіт з лабораторної роботи № 3

**Тема:** Дослідження векторних представлень слів української мови.

**Мета:** Дослідити Word2Vec, обчислити міри близькості, реалізувати
аналогії типу `країна1 - столиця1 + столиця2` та побудувати PCA-графіки.

## Хід виконання

1. Завантажено тестовий набір `country-capital` та модель Word2Vec (300 вимірів).
2. Реалізовано косинусну подібність і евклідову відстань.
3. Реалізовано пошук аналогій з виключенням слів із запиту.
4. Перевірено результати на {metrics['total']} прикладах.
5. Побудовано PCA-графіки обраних і всіх унікальних доступних слів.

## Кількісні результати

- Кількість тестів: **{metrics['total']}**
- Можливо обчислити (усі 4 слова є в моделі): **{metrics['evaluated']}**
- Правильних відповідей: **{metrics['correct']}**
- Точність за всіма тестами: **{metrics['accuracy_all']:.4f} ({metrics['accuracy_all']:.2%})**
- Точність серед доступних тестів: **{metrics['accuracy_covered']:.4f} ({metrics['accuracy_covered']:.2%})**
- Частка доступних тестів (coverage): **{metrics['coverage']:.2%}**
- Пропущено через відсутні вхідні слова: {metrics['missing_input']}
- Пропущено через відсутнє правильне слово: {metrics['missing_target']}
- Пропущено через нульовий вектор: {metrics['zero_query']}

## Приклади аналогій

| Столиця 1 | Країна 1 | Столиця 2 | Передбачена країна 2 | Косинусна схожість |
|---|---|---|---|---|
"""
    for ex in examples:
        report += f"| {ex['city1']} | {ex['country1']} | {ex['city2']} | {ex['answer']} | {ex['similarity']} |\n"
    report += """
## PCA-візуалізація

![PCA для обраних слів](pca_selected.png)

![PCA для унікальних слів тестового набору](pca_all.png)

**Висновок.** У роботі реалізовано векторні операції та пошук семантичних
аналогій для українських слів. Визначено точність на тестовому наборі;
частина помилок може бути пов'язана з неповним словником, неоднозначністю
слів і тим, що близькість векторів не гарантує правильної географічної
відповіді. PCA дозволяє наочно розглянути структуру векторів у двох вимірах.

*Важливо:* PCA-зображення є проєкцією та не відтворює всіх відстаней з
300-вимірного простору. Оцінку `accuracy_all` обчислено за всіма прикладами,
включно з тими, де слова відсутні (вони зараховані як помилки).
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    docs = BASE_DIR / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    with (docs / "results.json").open("w", encoding="utf-8") as handle:
        json.dump({"metrics": metrics, "selected_points": selected_points,
                   "all_points": all_points, "examples": examples},
                  handle, ensure_ascii=False, indent=2)


def run_experiment(source: str = "original", max_tests: int | None = None,
                   batch_size: int = 48, vocab_block: int = 40_000) -> dict:
    data_path = download_dataset()
    data = load_capital_analogies(data_path)
    if max_tests is not None:
        if max_tests <= 0:
            raise ValueError("max_tests має бути додатним числом")
        data = data.head(max_tests).copy()
    print(f"Зчитано рядків country-capital: {len(data)}", flush=True)
    model = load_model(obtain_model(source))

    print("Перевірка метричних функцій:")
    for a, b in [("Україна", "Польща"), ("Київ", "Варшава")]:
        if a in model and b in model:
            print(f"  {a} / {b}: cosine={cosine_similarity(model[a], model[b]):.4f}, "
                  f"euclidean={euclidean_distance(model[a], model[b]):.4f}")

    examples = []
    for city1, country1, city2 in [("Афіни", "Греція", "Київ"),
                                   ("Варшава", "Польща", "Берлін"),
                                   ("Берлін", "Німеччина", "Париж")]:
        found = get_country(city1, country1, city2, model)
        examples.append({"city1": city1, "country1": country1, "city2": city2,
                         "answer": found[0] if found else "немає в моделі",
                         "similarity": f"{found[1]:.4f}" if found else "—"})
        print(f"Аналогія {country1} - {city1} + {city2} -> {examples[-1]['answer']}")

    metrics, predictions = get_accuracy(model, data, batch_size=batch_size,
                                        vocab_block=vocab_block)
    print(f"Accuracy (усі): {metrics['accuracy_all']:.2%}; "
          f"accuracy (доступні): {metrics['accuracy_covered']:.2%}; "
          f"coverage: {metrics['coverage']:.2%}")
    out_dir = BASE_DIR / "results"
    selected_points = visualize(SELECTED_WORDS, model, out_dir / "pca_selected.png")
    unique_words = list(dict.fromkeys(str(w) for w in data.to_numpy().ravel()))
    all_points = visualize(unique_words, model, out_dir / "pca_all.png")
    save_results(metrics, predictions, selected_points, all_points, examples, out_dir)
    print("Готово! Результати збережено в results/ і docs/results.json", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР №3 — Word2Vec аналогії й PCA")
    parser.add_argument("--source", choices=["original", "huggingface"], default="original",
                        help="джерело моделі (за замовчуванням original; HF — резервне)")
    parser.add_argument("--max-tests", type=int, default=None,
                        help="для короткої перевірки, наприклад 50; без параметра — ВСІ 4271")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--vocab-block", type=int, default=40_000)
    args = parser.parse_args()
    run_experiment(args.source, args.max_tests, args.batch_size, args.vocab_block)


if __name__ == "__main__":
    main()
