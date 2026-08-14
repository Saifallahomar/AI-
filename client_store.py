import os
import uuid
import pandas as pd


LOCAL_DATA_DIR = "local_data"
CLIENTS_FILE = os.path.join(LOCAL_DATA_DIR, "verified_clients.csv")
COVERS_FILE = os.path.join(LOCAL_DATA_DIR, "verified_cover_lines.csv")


def save_verified_client(prospect):
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

    client_id = f"verified_{uuid.uuid4().hex[:8]}"

    # Save company profile
    profile = pd.DataFrame([{
        "client_id": client_id,
        "vertical": prospect.vertical,
        "employee_count": prospect.employee_count,
        "turnover_gbp": prospect.turnover_gbp,
        "funding_raised_gbp": prospect.funding_raised_gbp,
        "funding_series": prospect.funding_series,
    }])

    if os.path.exists(CLIENTS_FILE):
        profile.to_csv(CLIENTS_FILE, mode="a", header=False, index=False)
    else:
        profile.to_csv(CLIENTS_FILE, index=False)

    # Save ONLY covers that actually have a confirmed limit
    cover_rows = []

    for cover, limit in prospect.current_covers.items():
        if limit is not None:
            cover_rows.append({
                "client_id": client_id,
                "canonical_cover": cover,
                "limit_amount": limit,
            })

    if cover_rows:
        covers = pd.DataFrame(cover_rows)

        if os.path.exists(COVERS_FILE):
            covers.to_csv(COVERS_FILE, mode="a", header=False, index=False)
        else:
            covers.to_csv(COVERS_FILE, index=False)

    return client_id
