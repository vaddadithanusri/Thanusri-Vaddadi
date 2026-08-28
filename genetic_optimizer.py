import random
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = Path("dataset/processed/train.csv")

MODEL_DIR = Path("models")
RESULT_DIR = Path("results/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Candidate pool after statistical filtering
CANDIDATE_FEATURES = 2000

# Number of features GA is encouraged to select
TARGET_FEATURES = 500

# Genetic Algorithm parameters
POPULATION_SIZE = 12
GENERATIONS = 8
TOURNAMENT_SIZE = 3

CROSSOVER_RATE = 0.80
MUTATION_RATE = 0.02

# To keep GA fitness evaluation manageable
GA_TRAIN_SAMPLES = 20000
GA_VALIDATION_SAMPLES = 5000


# ============================================================
# GENETIC ALGORITHM
# ============================================================

class GeneticFeatureSelector:

    def __init__(
        self,
        X_train,
        y_train,
        X_validation,
        y_validation,
        random_state=42
    ):

        self.X_train = X_train
        self.y_train = np.asarray(y_train)

        self.X_validation = X_validation
        self.y_validation = np.asarray(y_validation)

        self.random_state = random_state

        random.seed(random_state)
        np.random.seed(random_state)

        self.n_features = X_train.shape[1]

        self.best_chromosome = None
        self.best_fitness = -np.inf

    # --------------------------------------------------------
    # Create chromosome
    # --------------------------------------------------------

    def create_chromosome(self):

        chromosome = np.zeros(
            self.n_features,
            dtype=np.int8
        )

        selected = min(
            TARGET_FEATURES,
            self.n_features
        )

        indices = np.random.choice(
            self.n_features,
            size=selected,
            replace=False
        )

        chromosome[indices] = 1

        return chromosome

    # --------------------------------------------------------
    # Initialize population
    # --------------------------------------------------------

    def initialize_population(self):

        return [
            self.create_chromosome()
            for _ in range(POPULATION_SIZE)
        ]

    # --------------------------------------------------------
    # Fitness function
    # --------------------------------------------------------

    def fitness(self, chromosome):

        selected_indices = np.flatnonzero(chromosome)

        if len(selected_indices) < 10:
            return 0.0

        X_train_selected = self.X_train[:, selected_indices]

        X_val_selected = self.X_validation[:, selected_indices]

        model = LinearSVC(
            C=1.0,
            random_state=self.random_state
        )

        model.fit(
            X_train_selected,
            self.y_train
        )

        predictions = model.predict(
            X_val_selected
        )

        f1 = f1_score(
            self.y_validation,
            predictions,
            zero_division=0
        )

        # Small penalty for very large feature subsets.
        # This encourages compact solutions.
        feature_penalty = (
            len(selected_indices) /
            self.n_features
        )

        fitness_value = (
            f1 - 0.02 * feature_penalty
        )

        return fitness_value

    # --------------------------------------------------------
    # Tournament selection
    # --------------------------------------------------------

    def tournament_selection(
        self,
        population,
        fitness_scores
    ):

        candidates = random.sample(
            range(len(population)),
            TOURNAMENT_SIZE
        )

        winner = max(
            candidates,
            key=lambda index: fitness_scores[index]
        )

        return population[winner].copy()

    # --------------------------------------------------------
    # Crossover
    # --------------------------------------------------------

    def crossover(
        self,
        parent1,
        parent2
    ):

        if random.random() > CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()

        point = random.randint(
            1,
            self.n_features - 1
        )

        child1 = np.concatenate(
            [
                parent1[:point],
                parent2[point:]
            ]
        )

        child2 = np.concatenate(
            [
                parent2[:point],
                parent1[point:]
            ]
        )

        return child1, child2

    # --------------------------------------------------------
    # Mutation
    # --------------------------------------------------------

    def mutate(self, chromosome):

        mutation_mask = (
            np.random.random(
                self.n_features
            ) < MUTATION_RATE
        )

        chromosome[mutation_mask] = (
            1 -
            chromosome[mutation_mask]
        )

        return chromosome

    # --------------------------------------------------------
    # Repair chromosome
    # --------------------------------------------------------

    def repair(self, chromosome):

        selected = np.flatnonzero(chromosome)

        target = min(
            TARGET_FEATURES,
            self.n_features
        )

        # Add features if too few are selected
        if len(selected) < target:

            available = np.flatnonzero(
                chromosome == 0
            )

            needed = target - len(selected)

            if len(available) > 0:

                additions = np.random.choice(
                    available,
                    size=min(
                        needed,
                        len(available)
                    ),
                    replace=False
                )

                chromosome[additions] = 1

        # Remove features if too many are selected
        elif len(selected) > target:

            remove_count = len(selected) - target

            removals = np.random.choice(
                selected,
                size=remove_count,
                replace=False
            )

            chromosome[removals] = 0

        return chromosome

    # --------------------------------------------------------
    # Run Genetic Algorithm
    # --------------------------------------------------------

    def run(self):

        print("\nInitializing Genetic Algorithm...")

        population = self.initialize_population()

        for generation in range(
            GENERATIONS
        ):

            print(
                f"\nGeneration "
                f"{generation + 1}/{GENERATIONS}"
            )

            fitness_scores = []

            for index, chromosome in enumerate(
                population
            ):

                score = self.fitness(
                    chromosome
                )

                fitness_scores.append(score)

                print(
                    f"  Individual "
                    f"{index + 1}/{len(population)} "
                    f"Fitness: {score:.5f} "
                    f"Features: "
                    f"{chromosome.sum()}"
                )

            best_index = int(
                np.argmax(fitness_scores)
            )

            generation_best = (
                fitness_scores[best_index]
            )

            if generation_best > self.best_fitness:

                self.best_fitness = (
                    generation_best
                )

                self.best_chromosome = (
                    population[best_index].copy()
                )

            print(
                f"Best generation fitness: "
                f"{generation_best:.5f}"
            )

            print(
                f"Overall best fitness: "
                f"{self.best_fitness:.5f}"
            )

            # ------------------------------------------------
            # Elitism
            # ------------------------------------------------

            new_population = [
                population[best_index].copy()
            ]

            # ------------------------------------------------
            # Generate new population
            # ------------------------------------------------

            while len(new_population) < POPULATION_SIZE:

                parent1 = self.tournament_selection(
                    population,
                    fitness_scores
                )

                parent2 = self.tournament_selection(
                    population,
                    fitness_scores
                )

                child1, child2 = self.crossover(
                    parent1,
                    parent2
                )

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                child1 = self.repair(child1)
                child2 = self.repair(child2)

                new_population.append(child1)

                if len(new_population) < POPULATION_SIZE:
                    new_population.append(child2)

            population = new_population

        print("\nGenetic Algorithm completed.")

        return self.best_chromosome


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_ga_optimization():

    print("=" * 70)
    print("SuicideWatchAI - GENETIC ALGORITHM OPTIMIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    df = pd.read_csv(
        TRAIN_PATH
    )

    texts = df["clean_text"].astype(str)
    labels = df["label"].astype(int)

    # --------------------------------------------------------
    # Load existing TF-IDF vectorizer
    # --------------------------------------------------------

    vectorizer_path = (
        MODEL_DIR /
        "tfidf_vectorizer.joblib"
    )

    print("\nLoading TF-IDF vectorizer...")

    vectorizer = joblib.load(
        vectorizer_path
    )

    print("Transforming training text...")

    X = vectorizer.transform(
        texts
    )

    y = labels.values

    print(
        f"Original TF-IDF matrix: "
        f"{X.shape}"
    )

    # --------------------------------------------------------
    # Candidate feature selection
    # --------------------------------------------------------

    print(
        f"\nSelecting top "
        f"{CANDIDATE_FEATURES:,} "
        f"candidate features..."
    )

    selector = SelectKBest(
        score_func=chi2,
        k=CANDIDATE_FEATURES
    )

    X_candidate = selector.fit_transform(
        X,
        y
    )

    print(
        f"Candidate matrix: "
        f"{X_candidate.shape}"
    )

    # --------------------------------------------------------
    # Create GA train/validation subset
    # --------------------------------------------------------

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    indices = np.arange(
        X_candidate.shape[0]
    )

    rng.shuffle(indices)

    train_count = min(
        GA_TRAIN_SAMPLES,
        len(indices)
    )

    validation_start = train_count

    validation_count = min(
        GA_VALIDATION_SAMPLES,
        len(indices) - validation_start
    )

    train_indices = indices[
        :train_count
    ]

    validation_indices = indices[
        validation_start:
        validation_start + validation_count
    ]

    X_ga_train = X_candidate[
        train_indices
    ]

    y_ga_train = y[
        train_indices
    ]

    X_ga_validation = X_candidate[
        validation_indices
    ]

    y_ga_validation = y[
        validation_indices
    ]

    print(
        f"\nGA training samples: "
        f"{len(train_indices):,}"
    )

    print(
        f"GA validation samples: "
        f"{len(validation_indices):,}"
    )

    # --------------------------------------------------------
    # Run GA
    # --------------------------------------------------------

    ga = GeneticFeatureSelector(
        X_ga_train,
        y_ga_train,
        X_ga_validation,
        y_ga_validation,
        random_state=RANDOM_STATE
    )

    best_chromosome = ga.run()

    # --------------------------------------------------------
    # Save optimized feature information
    # --------------------------------------------------------

    selected_candidate_indices = (
        np.flatnonzero(
            best_chromosome
        )
    )

    original_feature_indices = (
        selector.get_support(
            indices=True
        )
    )

    selected_original_indices = (
        original_feature_indices[
            selected_candidate_indices
        ]
    )

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    selected_feature_names = (
        feature_names[
            selected_original_indices
        ]
    )

    np.save(
        MODEL_DIR /
        "ga_selected_feature_indices.npy",
        selected_original_indices
    )

    np.save(
        MODEL_DIR /
        "ga_selected_feature_names.npy",
        selected_feature_names
    )

    joblib.dump(
        selector,
        MODEL_DIR /
        "candidate_feature_selector.joblib"
    )

    print(
        f"\nGA selected "
        f"{len(selected_feature_names):,} "
        f"features."
    )

    print("\nFirst 30 selected features:")

    for feature in selected_feature_names[:30]:
        print(
            f"  {feature}"
        )

    print("\n" + "=" * 70)
    print("GENETIC ALGORITHM FEATURE SELECTION COMPLETE")
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_ga_optimization()