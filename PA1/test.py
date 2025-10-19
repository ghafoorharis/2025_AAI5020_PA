import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from glob import glob
from model import NetVLADModel
from eval import compute_similarity_matrices
# Newly defined libraries should be listed here.
import cv2

class IndoorDataset(Dataset):
    def __init__(self, base_dir, transform=None, scenes=None):
        self.items = []
        if scenes is None:
            scenes = ["P001","P002","P003","P004"]
        for scene in scenes:
            img_dir = os.path.join(base_dir, scene, "images")
            pose_file = os.path.join(base_dir, scene, "pose.txt")
            images = sorted(glob(os.path.join(img_dir, "*.png")))
            poses = np.loadtxt(pose_file).reshape(-1, 7)
            n = min(len(images), len(poses))
            for idx in range(0, n):
                self.items.append((images[idx], poses[idx], scene, idx))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, pose, scene, frame_idx = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, pose.astype(np.float32), img_path, scene, frame_idx

@torch.no_grad()
def extract_embeddings(model, loader, device):
    all_vecs, all_meta = [], []
    for imgs, poses, paths, scenes, idxs in loader:
        imgs = imgs.to(device)
        vecs = model(imgs).cpu()
        all_vecs.append(vecs)
        for i in range(len(paths)):
            p = poses[i].numpy() if hasattr(poses[i], "numpy") else np.asarray(poses[i])
            all_meta.append({
                "path": paths[i],
                "pose": p,
                "scene": scenes[i],
                "idx": int(idxs[i]),
            })
    return torch.cat(all_vecs, dim=0), all_meta

def pose_error(pose_est, pose_gt):
    p1 = np.asarray(pose_est, dtype=np.float64)
    p2 = np.asarray(pose_gt, dtype=np.float64)
    t_err = float(np.linalg.norm(p1[:3] - p2[:3]))
    q1, q2 = p1[3:7].copy(), p2[3:7].copy()
    n1, n2 = np.linalg.norm(q1), np.linalg.norm(q2)
    if n1 == 0 or n2 == 0:
        r_err = np.nan
    else:
        q1 /= n1; q2 /= n2
        dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
        r_err = float(2.0 * np.degrees(np.arccos(dot)))
    return t_err, r_err

# minimal quaternion helpers
def _q_norm(q):
    """
    Normalize a quaternion
    Args:
        q: Quaternion
    Returns:
        q: Normalized quaternion
    """
    n = np.linalg.norm(q)
    return q if n == 0 else (q / n)

def _q_to_R(q):  # [qw,qx,qy,qz] -> Rcw
    '''
    Convert a quaternion to a rotation matrix
    Ref: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    Args:
        q: Quaternion
    Returns:
        R: Rotation matrix
    '''
    qw, qx, qy, qz = _q_norm(q.astype(np.float64))
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)

def _R_to_q(R):
    '''
   Convert a rotation matrix to a quaternion
   Ref: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
   Args:
       R: Rotation matrix
   Returns:
       q: Quaternion
    '''
    R = R.astype(np.float64)
    tr = float(np.trace(R))
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S; qx = 0.25*S
            qy = (R[0,1] + R[1,0]) / S; qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S; qy = 0.25*S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S; qz = 0.25*S
    return _q_norm(np.array([qw,qx,qy,qz], dtype=np.float64))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NetVLADModel(num_clusters=16).to(device)
    ckpt_path = "./checkpoints/netvlad_final.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    base_dir = "./dataset/camera"
    full_dataset = IndoorDataset(base_dir, transform=transform, scenes=["P001","P002","P003","P004"])
    n_total = len(full_dataset)

    n_query = max(1, int(0.2 * n_total))
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=g).tolist()

    query_indices = perm[:n_query]
    index_indices = perm[n_query:]

    query_set = Subset(full_dataset, query_indices)
    index_set  = Subset(full_dataset, index_indices)

    print(f"Total: {n_total} | Query: {len(query_set)} | Index: {len(index_set)}")

    query_loader = DataLoader(query_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    index_loader = DataLoader(index_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    q_vecs, q_meta = extract_embeddings(model, query_loader, device)
    db_vecs, db_meta = extract_embeddings(model, index_loader, device)


    # -------------------------------
    # TASK4 : Camera Pose Estimation 
    # -------------------------------
    '''
    1. Compute the distance matrix between the query and database embeddings
    2. Find the best database index for each query
    3. Find the Features and Descriptors for the query and database images
    4. Find the Essential Matrix using initial camera intrinsics 
    5. Recover the relative pose between the query and database images

    Reference: https://rbe549.github.io/spring2023/proj/p2/#due
    '''
    dist_mat = torch.cdist(q_vecs, db_vecs, p=2)
    best_dists, best_db_idxs = dist_mat.min(dim=1) # for each query vector, find the minimum distance to any database vector

    rows = []
    for qi in range(q_vecs.shape[0]):
        q_i = q_meta[qi]
        q_img = cv2.imread(q_i["path"], cv2.IMREAD_GRAYSCALE)

        db_i = db_meta[int(best_db_idxs[qi].item())]
        db_img = cv2.imread(db_i["path"], cv2.IMREAD_GRAYSCALE)
        best_dist_val = float(best_dists[qi].item())

        # default: copy DB pose (fallback if geometry fails)
        estimated_pose = db_i["pose"].copy()

        det = cv2.SIFT_create(nfeatures=100)
        norm = cv2.NORM_L2
        kp1, des1 = det.detectAndCompute(q_img,  None)  # query first
        kp2, des2 = det.detectAndCompute(db_img, None)  # db second
        bf = cv2.BFMatcher(norm, crossCheck=False)
        knn12 = bf.knnMatch(des1, des2, k=2)  # q->db
        good = []
        for mn in knn12:
            if len(mn) < 2: continue
            m, n = mn
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) >= 8:
            pts_q  = np.float32([kp1[m.queryIdx].pt for m in good])
            pts_db = np.float32([kp2[m.trainIdx].pt for m in good])

            # Essential with (q, db) so R_rel is q->db
            h, w = q_img.shape[:2]
            f = float(max(h, w))
            pp = (w * 0.5, h * 0.5)
            E, _ = cv2.findEssentialMat(pts_q, pts_db, focal=f, pp=pp,
                                        method=cv2.RANSAC, prob=0.999, threshold=0.7)
            if E is not None:
                inl, R_rel, t_rel, _ = cv2.recoverPose(E, pts_q, pts_db, focal=f, pp=pp)
                if inl is not None and int(inl) >= 20:
                    # Compose absolute rotation: Rcw_q = Rcw_d @ R_rel  (R_rel: q->db)
                    Rcw_d = _q_to_R(db_i["pose"][3:7])
                    Rcw_q = Rcw_d @ R_rel
                    # Keep translation same as DB (scale unknown with 1 pair)
                    C_q = db_i["pose"][:3]
                    q_q = _R_to_q(Rcw_q)
                    estimated_pose = np.concatenate([C_q, q_q], axis=0)

        t_err, r_err = pose_error(estimated_pose, q_i["pose"])
        rows.append({
            "q_path": q_i["path"],
            "db_path": db_i["path"],
            "d_embed": best_dist_val,
            "t_err": float(t_err),
            "r_err": float(r_err),
        })

    # -------------------------------

    with open("output/result_pose_eval.txt", "w") as f:
        f.write("Top-1 retrieval pose evaluation\n")
        f.write(f"(queries={len(rows)}, index={len(db_meta)})\n\n")
        for r in rows:
            f.write(f"Query: {r['q_path']}\n")
            f.write(f"Best Match: {r['db_path']}\n")
            f.write(f"Embed Dist: {r['d_embed']:.4f}\n")
            f.write(f"Translation Error: {r['t_err']:.4f} m, Rotation Error: {r['r_err']:.2f} deg\n\n")

    mean_t = float(np.nanmean([r["t_err"] for r in rows]))
    mean_r = float(np.nanmean([r["r_err"] for r in rows]))
    print(f"Mean Pose Error: {mean_t:.4f}")
    print(f"Mean Rotation Error: {mean_r:.2f} deg")
