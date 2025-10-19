import torch
import numpy as np
from tqdm import tqdm

# Optional libraries for visualization
try:
    import matplotlib.pyplot as plt
    from PIL import Image
    import os

    flag = True
except ImportError:
    print(
        "Optional libraries for visualization are not installed. Please install them using: pip install matplotlib pillow"
    )
    flag = False


def compute_embeddings(
    model, dataset, device, batch_size=32
) -> tuple[np.ndarray, list, list]:
    """
    Computes the embeddings for a given dataset using the model.
    Args:
        model: The model to use for embedding computation.
        dataset: The dataset to compute embeddings for.
        device: The device to use for computation.
        batch_size: The batch size to use for computation.
    Returns:
        embeddings: The embeddings for the dataset.
        indices: The indices of the dataset.
        paths: The paths of the dataset.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings, indices, paths = [], [], []

    model.eval()
    with torch.no_grad():
        for imgs, idxs, ps in tqdm(loader, desc="Extracting embeddings"):
            imgs = imgs.to(device)
            embs = model(imgs).cpu().numpy()
            embeddings.append(embs)
            indices.extend(idxs.numpy().astype(int))
            paths.extend(ps)

    embeddings = np.vstack(embeddings)
    return embeddings, indices, paths


def get_co_ord_dist(q_utm, db_utm):
    """
    Computes the co-ordinate distance between two points on the earth's surface using L2 norm
    Args:
        q_utm (tuple): The UTM coordinates of the query point (latitude, longitude)
        db_utm (tuple): The UTM coordinates of the database point (latitude, longitude)
    Returns:
        float: The co-ordinate distance between the two points in meters
    """
    x_lat, x_long = q_utm
    y_lat, y_long = db_utm
    co_ord_dist = np.sqrt((x_long - y_long) ** 2 + (x_lat - y_lat) ** 2)
    return co_ord_dist


def compute_similarity_matrices(
    q_embs_dict: dict, db_embs_dict: dict, find_gt_dist: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute similarity scores and ground truth distance matrices

    Args:
        q_embs_dict: Dictionary of query embeddings
        db_embs_dict: Dictionary of database embeddings
        find_gt_dist: Whether to find the ground truth distance
    Returns:
        sim_scores: Similarity scores matrix
        ground_truth_dist: Ground truth distance matrix
    """
    n_queries = len(q_embs_dict)
    n_db = len(db_embs_dict)
    # create similarity scores and ground truth distance matrices
    sim_scores = np.zeros((n_queries, n_db))
    ground_truth_dist = np.zeros((n_queries, n_db))

    for query_idx in range(n_queries):
        for db_idx in range(n_db):
            cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)(
                torch.tensor(q_embs_dict[query_idx]["embs"]),
                torch.tensor(db_embs_dict[db_idx]["embs"]),
            )
            if find_gt_dist:
                cord_dist = get_co_ord_dist(
                    q_embs_dict[query_idx]["utm"],  # query UTM coordinates
                    db_embs_dict[db_idx]["utm"],  # database UTM coordinates
                )
                ground_truth_dist[query_idx, db_idx] = (
                    cord_dist  # co-ordinate L2 - norm distance between query and database images
                )

            sim_scores[query_idx, db_idx] = (
                cos_sim.detach().numpy().item()
            )  # cosine similarity between query and database embeddings

    return sim_scores, ground_truth_dist


def evaluate_recall(
    model,
    query_ds,
    db_ds,
    utmQ,
    utmDb,
    posDistThr,
    device,
    recall_values=[1, 5, 10],
    preview=False,
    save_dir="output/visualizations",
    num_examples=5,
):
    """
    Complete evaluation function with recall calculation and visualization

    Args:
        model: The model to use for evaluation
        query_ds: The query dataset
        db_ds: The database dataset
        utmQ: The UTM coordinates of the query images
        utmDb: The UTM coordinates of the database images
        posDistThr: The position distance threshold
        device: The device to use for evaluation
        recall_values: The recall values to evaluate
        preview: Whether to preview the results
        save_dir: The directory to save the results
        num_examples: Number of query examples to visualize
    Returns:
        recalls: The recall values
        sim_scores: The similarity scores matrix
        ground_truth_dist: The ground truth distance matrix
    """
    # Compute embeddings
    q_embeddings, q_indices, q_paths = compute_embeddings(model, query_ds, device)
    db_embeddings, db_indices, db_paths = compute_embeddings(model, db_ds, device)

    # Create dictionaries
    q_embs_dict = {}
    for x, y, z in zip(q_indices, q_embeddings, q_paths):
        q_embs_dict[x] = {"embs": y, "path": z, "utm": utmQ[x], "th": posDistThr}

    db_embs_dict = {}
    for x, y, z in zip(db_indices, db_embeddings, db_paths):
        db_embs_dict[x] = {"embs": y, "path": z, "utm": utmDb[x]}

    # Compute similarity matrices
    sim_scores, ground_truth_dist = compute_similarity_matrices(
        q_embs_dict=q_embs_dict,
        db_embs_dict=db_embs_dict,
        find_gt_dist=True,
    )

    # Calculate recall@K
    recalls = {k: 0 for k in recall_values}
    n_queries = len(q_embs_dict)

    for query_idx in range(n_queries):
        # Get ground truth positives
        # threshold for query image
        threshold = q_embs_dict[query_idx]["th"]
        positives = np.where(ground_truth_dist[query_idx] < threshold)[0]
        if len(positives) == 0:
            # Fallback to closest image if no positives
            positives = [np.argmin(ground_truth_dist[query_idx])]

        # Get top predictions for each recall value
        max_k = max(recall_values)
        top_indices = np.argsort(sim_scores[query_idx])[-max_k:][
            ::-1
        ]  # All Similar DB Images sorted by similarity score [MAX K Images]

        for (
            k
        ) in (
            recall_values
        ):  # for each recall value, check if atleast one positive in top k predictions
            top_k_indices = top_indices[:k]  # top k predictions for each recall value
            if np.any(
                np.in1d(top_k_indices, positives)
            ):  # atleast one positive in top k predictions
                recalls[k] += 1  # increment the recall value

    # Convert to percentages
    # Print results
    print("\n" + "=" * 50)
    print("RECALL@K RESULTS")
    print("=" * 50)
    for k in recall_values:
        hits = recalls[k]
        recalls[k] = (hits / n_queries) * 100  # convert to percentage
        print(f"Recall@{k}: {hits}/{n_queries} = {recalls[k]:.2f}%")

    # Lets create the plot for recall values

    plt.figure(figsize=(10, 5))
    plt.plot(recall_values, list(recalls.values()), marker="o", color="blue")
    plt.xlabel("Recall@N", fontsize=14)
    plt.ylabel("Recall Value", fontsize=14)
    plt.title(f"Recall@N Plot for {save_dir}", fontsize=16)
    plt.xticks(recall_values, fontsize=12)
    plt.yticks(list(recalls.values()), fontsize=12)
    plt.savefig(os.path.join("output", "task3_result_graph.png"))
    plt.close()

    # Visualization if requested and optional libraries are installed
    if flag and preview:
        os.makedirs(save_dir, exist_ok=True)
        visualize_retrieval_results(
            q_embs_dict,
            db_embs_dict,
            sim_scores,
            ground_truth_dist,
            posDistThr,
            save_dir,
            num_examples=num_examples,
            k=5,
        )

    return recalls, sim_scores, ground_truth_dist


def visualize_retrieval_results(
    q_embs_dict: dict,
    db_embs_dict: dict,
    sim_scores: np.ndarray,
    ground_truth_dist: np.ndarray,
    posDistThr: float,
    save_dir: str,
    num_examples: int = 5,
    k: int = 5,
):
    """
    Visualize retrieval results for random query examples
    Args:
        q_embs_dict: Dictionary of query embeddings
        db_embs_dict: Dictionary of database embeddings
        sim_scores: Similarity scores matrix
        ground_truth_dist: Ground truth distance matrix
        posDistThr: Position distance threshold
        save_dir: Directory to save the visualizations
        num_examples: Number of query examples to visualize
        k: Number of top predictions to visualize
    """
    n_queries = len(q_embs_dict)
    query_indices = np.random.choice(n_queries, num_examples, replace=False)

    for count, query_idx in enumerate(query_indices):
        # Get query info
        ground_truth_threshold = posDistThr

        # Get top k predictions
        top_k_indices = np.argsort(sim_scores[query_idx])[-k:][::-1]
        top_k_db_images = [db_embs_dict[idx]["path"] for idx in top_k_indices]
        top_k_db_images_ground_truth_dist = [
            ground_truth_dist[query_idx, idx] for idx in top_k_indices
        ]

        # Get ground truth positives
        ground_truth_positive_indices = np.where(
            ground_truth_dist[query_idx] < ground_truth_threshold
        )[0]
        ground_truth_positive_images = [
            db_embs_dict[idx]["path"] for idx in ground_truth_positive_indices
        ]

        # Get closest ground truth match
        if len(ground_truth_positive_indices) > 0:
            closest_gt_idx = ground_truth_positive_indices[
                np.argmin(ground_truth_dist[query_idx, ground_truth_positive_indices])
            ]
            closest_gt_path = db_embs_dict[closest_gt_idx]["path"]
            closest_gt_dist = ground_truth_dist[query_idx, closest_gt_idx]
            closest_gt_sim = sim_scores[query_idx, closest_gt_idx]
        else:
            closest_gt_idx = None

        # Count true positives in top k
        true_positives = np.sum(
            np.array(top_k_db_images_ground_truth_dist) < ground_truth_threshold
        )

        # Create visualization
        total_subplots = k + 2 if closest_gt_idx is not None else k + 1
        fig = plt.figure(figsize=(20, 5))

        # Plot query image
        plt.subplot(1, total_subplots, 1)
        query_img_path = q_embs_dict[query_idx]["path"]
        query_img = Image.open(query_img_path).convert("RGB")
        plt.imshow(query_img)
        plt.axis("off")
        plt.title(
            f"Query Image\nID: {query_idx}\nThreshold: {ground_truth_threshold:.2f}",
            fontsize=10,
            weight="bold",
        )

        # Sort top-k by true positives first
        sorted_indices_with_db_idx = list(
            zip(top_k_indices, top_k_db_images_ground_truth_dist)
        )
        sorted_indices_with_db_idx.sort(
            key=lambda x: (
                ground_truth_dist[query_idx, x[0]] >= ground_truth_threshold,
                -sim_scores[query_idx, x[0]],
            )
        )

        # Plot top-k database images
        for i, (db_idx, gt_dist) in enumerate(sorted_indices_with_db_idx):
            plt.subplot(1, total_subplots, i + 2)

            # Load and display database image
            db_img_path = db_embs_dict[db_idx]["path"]
            db_img = Image.open(db_img_path).convert("RGB")
            plt.imshow(db_img)

            # Add colored border
            if gt_dist < ground_truth_threshold:
                border_color = "green"
                border_label = "TP"
            else:
                border_color = "red"
                border_label = "FP"

            plt.gca().add_patch(
                plt.Rectangle(
                    (0, 0),
                    db_img.width,
                    db_img.height,
                    fill=False,
                    edgecolor=border_color,
                    linewidth=4,
                )
            )
            plt.axis("off")

            title = (
                f"Rank {i+1} [{border_label}]\n"
                f"Cos Sim: {sim_scores[query_idx, db_idx]:.3f}\n"
                f"GT Dist: {gt_dist:.2f}"
            )
            plt.title(title, fontsize=9, color=border_color, weight="bold")

        # Plot closest ground truth match if exists
        if closest_gt_idx is not None:
            plt.subplot(1, total_subplots, total_subplots)
            gt_img_path = closest_gt_path
            gt_img = Image.open(gt_img_path).convert("RGB")
            plt.imshow(gt_img)

            plt.gca().add_patch(
                plt.Rectangle(
                    (0, 0),
                    gt_img.width,
                    gt_img.height,
                    fill=False,
                    edgecolor="blue",
                    linewidth=4,
                )
            )
            plt.axis("off")

            in_top_k = closest_gt_idx in top_k_indices
            top_k_status = f" (In Top-{k})" if in_top_k else f" (Not in Top-{k})"

            gt_title = (
                f"Closest Ground Truth{top_k_status}\n"
                f"Cos Sim: {closest_gt_sim:.3f}\n"
                f"GT Dist: {closest_gt_dist:.2f}"
            )
            plt.title(gt_title, fontsize=9, color="blue", weight="bold")

        # Main title and save
        query_filename = os.path.basename(q_embs_dict[query_idx]["path"])
        plt.suptitle(
            f"Query: {query_filename}\n"
            f"True Positives in Top-{k}: {true_positives}/{k} | "
            f"Total GT Positives: {len(ground_truth_positive_images)}",
            fontsize=12,
            weight="bold",
            y=1.05,
        )
        plt.tight_layout()

        # Save figure with proper name
        save_path = os.path.join(save_dir, f"query_{query_idx}_retrieval.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        # print(f"Saved visualization: {save_path}")
        plt.close()

        # # Print detailed results
        # print(f"\nQuery {query_idx} Results:")
        # print(f"  Query path: {q_embs_dict[query_idx]['path']}")
        # print(f"  True positives in top-{k}: {true_positives}/{k}")
        # print(f"  Total ground truth positives: {len(ground_truth_positive_images)}")
        # if closest_gt_idx is not None:
        #     print(f"  Closest GT in top-{k}: {'Yes' if in_top_k else 'No'}")

    print(f"\nVisualized {num_examples} query examples. Figures saved to: {save_dir}")
