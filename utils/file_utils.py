# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

# coding: utf-8
import csv
import json
import os
from pprint import pprint

import torch


def save_alpha(path, alpha_array):
    # precision -> [epoch, acc_1, ..., acc_i, acc_mean]
    header = ["Iterations"] + [f"alpha_{i}" for i in range(len(alpha_array[0]) - 1)]
    with open(path + '/alpha.csv', 'w', newline='') as csvFile:
        csvwriter = csv.writer(csvFile)
        csvwriter.writerow(header)
        csvwriter.writerows(alpha_array)


def save_weight_scores(weight_scores, path, learners):
    # column_names = [f"Client {i}" for i in range(len(weight_scores[0]))]
    column_names = [f"{learners[i].data_type} - {i}" for i in range(len(learners))]
    with open(path + '/pond_score.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(column_names)
        writer.writerows(weight_scores)


def save_accuracies(path, accuracy_array, learners=None):
    if learners is None:
        header = ["Iterations", "acc"]
    else:
        header = ["Iterations"] + [f"{learners[i].data_type} - {i}" for i in range(len(learners))]
    with open(path + '/acc.csv', 'w', newline='') as csvFile:
        csvwriter = csv.writer(csvFile)
        csvwriter.writerow(header)
        csvwriter.writerows(accuracy_array)


def save_accuracies_delta(path, accuracy_array, learners, name='acc_delta'):
    # acc_i --> [acc_pre, acc_post]
    # accuracy --> [epoch, acc_1, ..., acc_i]
    header = ["Iterations"] + [f"{learners[i].data_type} - {i}" for i in range(len(learners))]
    with open(path + f'/{name}.csv', 'w', newline='') as csvFile:
        csvwriter = csv.writer(csvFile)
        csvwriter.writerow(header)
        csvwriter.writerows(accuracy_array)


def save_scores(mal_score, path, learners):
    column_names = [f"{learners[i].data_type} - {i}" for i in range(len(learners))]
    with open(path + '/score.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(column_names)
        writer.writerows(mal_score)


def print_config(c):
    with open(os.path.join(c.PATH, 'config.txt'), 'w') as f:
        pprint(vars(c), stream=f)


def save_clients(path, learners=None):
    """
    Save only what's necessary for later evaluation:
    - local model state_dict
    - client metadata (data_type, client_id, test_indices)
    - optionally the global model
    """
    clients_dir = os.path.join(path, "clients")
    os.makedirs(clients_dir, exist_ok=True)

    # Save each client
    for idx, learner in enumerate(learners):
        client_id = f"{idx}_{learner.data_type}" if hasattr(learner, "data_type") else str(idx)

        # Save model
        model_path = os.path.join(clients_dir, f"client_{client_id}.pt")
        torch.save(learner.trainer.model.state_dict(), model_path)

        # Save metadata
        metadata = {
            "client_id": client_id,
            "data_type": getattr(learner, "data_type", None),
            "rank": learner.c.RANK
        }
        if hasattr(learner, "test_indices"):
            metadata["test_indices"] = list(learner.test_indices)

        with open(os.path.join(clients_dir, f"client_{client_id}_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"[INFO] Saved {len(learners)} clients in: {clients_dir}")


def save_clients_global(path, learners=None):
    """
    Case where all clients share the same global model: FedAvg, FedProx, SCAFFOLD
    """
    clients_dir = os.path.join(path, "clients")
    os.makedirs(clients_dir, exist_ok=True)

    # Save each client
    for idx, learner in enumerate(learners):
        client_id = f"{idx}_{learner.data_type}" if hasattr(learner, "data_type") else str(idx)

        # Save model
        if idx == 0:
            # Save global model only once
            model_path = os.path.join(clients_dir, f"global_model.pt")
            torch.save(learner.trainer.model.state_dict(), model_path)

        # Save metadata
        metadata = {
            "client_id": client_id,
            "data_type": getattr(learner, "data_type", None),
            "rank": learner.c.RANK
        }
        if hasattr(learner, "test_indices"):
            metadata["test_indices"] = list(learner.test_indices)

        with open(os.path.join(clients_dir, f"client_{client_id}_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"[INFO] Saved {len(learners)} clients in: {clients_dir}")


def save_clients_flprotector(path, learners=None, metrics=None):
    """
    Save only what's necessary for later evaluation:
    - local model state_dict
    - client metadata (data_type, client_id, test_indices)
    - optionally the global model
    """
    clients_dir = os.path.join(path, "clients")
    os.makedirs(clients_dir, exist_ok=True)

    # Save each client
    for idx, learner in enumerate(learners):
        client_id = f"{idx}_{learner.data_type}" if hasattr(learner, "data_type") else str(idx)

        # Save models
        model_path = os.path.join(clients_dir, f"global_model.pt")
        torch.save(learner.trainer.model.state_dict(), model_path)
        learner.add_personalization()
        model_path = os.path.join(clients_dir, f"client_{client_id}.pt")
        torch.save(learner.trainer.model.state_dict(), model_path)

        # Save encoder model if present
        if hasattr(learner.trainer, 'encoder'):
            encoder_path = os.path.join(clients_dir, f"client_{client_id}_encoder.pt")
            torch.save(learner.trainer.encoder.state_dict(), encoder_path)

        # Save metadata
        metadata = {
            "client_id": client_id,
            "data_type": getattr(learner, "data_type", None),
            "rank": learner.c.RANK,
            "mean_encoder_train": learner.trainer.mean_encoder_train.tolist() if hasattr(learner.trainer,
                                                                                         'encoder') else None,
            "std_encoder_train": learner.trainer.std_encoder_train.tolist() if hasattr(learner.trainer,
                                                                                       'encoder') else None
        }
        if metrics is not None and isinstance(metrics, dict):
            client_metrics = None
            clients_metrics = metrics.get("clients") if hasattr(metrics, "get") else None
            if clients_metrics:
                client_metrics = clients_metrics.get(str(idx)) or clients_metrics.get(client_id)
            if client_metrics:
                evaluation = {
                    "total_samples": int(client_metrics.get("total_samples", 0)),
                    "testing_accuracy": float(client_metrics.get("testing_accuracy", 0.0)),
                    "testing_correct": int(client_metrics.get("testing_correct", 0)),
                    "real_accuracy": float(client_metrics.get("real_accuracy", 0.0)),
                    "real_correct": int(client_metrics.get("real_correct", 0)),
                    "possible_best_accuracy": float(client_metrics.get("possible_best_accuracy", 0.0)),
                    "possible_best_correct": int(client_metrics.get("possible_best_correct", 0)),
                }
                if "selection_accuracy" in client_metrics:
                    evaluation["selection_accuracy"] = float(client_metrics["selection_accuracy"])
                if "selection_correct" in client_metrics:
                    evaluation["selection_correct"] = int(client_metrics["selection_correct"])
                if "selection_total" in client_metrics:
                    evaluation["selection_total"] = int(client_metrics["selection_total"])
                if "encoder_threshold" in client_metrics:
                    evaluation["encoder_threshold"] = float(client_metrics["encoder_threshold"])
                metadata["evaluation"] = evaluation
        if hasattr(learner, "test_indices"):
            metadata["test_indices"] = list(learner.test_indices)

        with open(os.path.join(clients_dir, f"client_{client_id}_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"[INFO] Saved {len(learners)} clients in: {clients_dir}")


def save_flprotector_metrics(path, metrics, filename_prefix="flprotector_eval", fmt="csv"):
    os.makedirs(path, exist_ok=True)

    if fmt == "both":
        do_csv = do_json = True
    elif fmt == "csv":
        do_csv, do_json = True, False
    elif fmt == "json":
        do_csv, do_json = False, True
    else:
        raise ValueError("formats must be 'csv', 'json' or 'both'")

    clients = metrics.get("clients", {}) if isinstance(metrics, dict) else {}
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    mode = metrics.get("mode") if isinstance(metrics, dict) else None

    fieldnames = [
        "client_id",
        "data_type",
        "total_samples",
        "testing_accuracy",
        "testing_correct",
        "real_accuracy",
        "real_correct",
        "possible_best_accuracy",
        "possible_best_correct",
    ]
    include_selection = any("selection_accuracy" in entry for entry in clients.values())
    if include_selection:
        fieldnames.extend(["selection_accuracy", "selection_correct", "selection_total"])
    if any("encoder_threshold" in entry for entry in clients.values()):
        fieldnames.append("encoder_threshold")

    if do_csv:
        csv_name = f"{filename_prefix}.csv" if mode is None else f"{filename_prefix}_{mode}.csv"
        csv_path = os.path.join(path, csv_name)
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for key in sorted(clients, key=lambda x: int(x) if str(x).isdigit() else str(x)):
                entry = clients[key]
                row = {
                    "client_id": entry.get("client_id", key),
                    "data_type": entry.get("data_type"),
                    "total_samples": int(entry.get("total_samples", 0)),
                    "testing_accuracy": float(entry.get("testing_accuracy", 0.0)),
                    "testing_correct": int(entry.get("testing_correct", 0)),
                    "real_accuracy": float(entry.get("real_accuracy", 0.0)),
                    "real_correct": int(entry.get("real_correct", 0)),
                    "possible_best_accuracy": float(entry.get("possible_best_accuracy", 0.0)),
                    "possible_best_correct": int(entry.get("possible_best_correct", 0)),
                }
                if include_selection:
                    row["selection_accuracy"] = (
                        float(entry.get("selection_accuracy", 0.0)) if "selection_accuracy" in entry else None
                    )
                    row["selection_correct"] = (
                        int(entry.get("selection_correct", 0)) if "selection_correct" in entry else None
                    )
                    row["selection_total"] = (
                        int(entry.get("selection_total", 0)) if "selection_total" in entry else None
                    )
                if "encoder_threshold" in entry:
                    row["encoder_threshold"] = float(entry.get("encoder_threshold"))
                writer.writerow(row)

            if overall:
                overall_row = {
                    "client_id": "overall",
                    "data_type": None,
                    "total_samples": int(overall.get("total_samples", 0)),
                    "testing_accuracy": float(overall.get("testing_accuracy", 0.0)),
                    "testing_correct": int(overall.get("testing_correct", 0)),
                    "real_accuracy": float(overall.get("real_accuracy", 0.0)),
                    "real_correct": int(overall.get("real_correct", 0)),
                    "possible_best_accuracy": float(overall.get("possible_best_accuracy", 0.0)),
                    "possible_best_correct": int(overall.get("possible_best_correct", 0)),
                }
                if include_selection:
                    overall_row["selection_accuracy"] = (
                        float(overall.get("selection_accuracy", 0.0)) if "selection_accuracy" in overall else None
                    )
                    overall_row["selection_correct"] = (
                        int(overall.get("selection_correct", 0)) if "selection_correct" in overall else None
                    )
                    overall_row["selection_total"] = (
                        int(overall.get("selection_total", 0)) if "selection_total" in overall else None
                    )
                writer.writerow(overall_row)

    if do_json:
        json_name = f"{filename_prefix}.json" if mode is None else f"{filename_prefix}_{mode}.json"
        json_path = os.path.join(path, json_name)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)


def save_clients_apfl(path, learners=None):
    """
    Save only what's necessary for later evaluation:
    - local model state_dict (delta_tilde)
    - client metadata (data_type, client_id, test_indices)
    - optionally the global model
    """
    clients_dir = os.path.join(path, "clients")
    os.makedirs(clients_dir, exist_ok=True)

    # Save each client
    for idx, learner in enumerate(learners):
        client_id = f"{idx}_{learner.data_type}" if hasattr(learner, "data_type") else str(idx)

        # Save models
        model_path = os.path.join(clients_dir, f"global_model.pt")
        torch.save(learner.trainer.model.state_dict(), model_path)
        model_path = os.path.join(clients_dir, f"client_{client_id}.pt")
        torch.save(learner.delta_tilde.state_dict(), model_path)

        # Save metadata
        metadata = {
            "client_id": client_id,
            "data_type": getattr(learner, "data_type", None),
            "rank": learner.c.RANK,
        }
        if hasattr(learner, "test_indices"):
            metadata["test_indices"] = list(learner.test_indices)

        with open(os.path.join(clients_dir, f"client_{client_id}_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"[INFO] Saved {len(learners)} clients in: {clients_dir}")
