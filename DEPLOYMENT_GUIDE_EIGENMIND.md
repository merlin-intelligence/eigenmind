# Eigenmind Production Deployment Guide

This guide covers the end-to-end deployment of Eigenmind on a **Google Cloud Compute Engine VM**, organised in two parts:

- **Part I — Initialisation**: one-shot setup performed once when standing up a new VM (§1 → §6).
- **Part II — Updates**: keeping the running VM in sync with the `main` branch of the repository (§7 → §8).

> The instructions below assume you have a Google Cloud account with billing enabled and the `gcloud` CLI installed locally. All `gcloud` commands can also be executed from the [Google Cloud Console](https://console.cloud.google.com) UI if preferred.

---

# Part I — Initialisation

This part is executed **once**, when provisioning a fresh VM. At the end of it you will have a Streamlit service running under `systemd`, listening on port 8501 of the VM's public IP.

## 1. Google Cloud Project Setup

### 1.1 Create (or select) a project
```bash
# Create a new project
gcloud projects create eigenmind-prod --name="Eigenmind Production"

# Set it as the active project
gcloud config set project eigenmind-prod
```

### 1.2 Link a billing account
```bash
# List your billing accounts
gcloud billing accounts list

# Link one to the project
gcloud billing projects link eigenmind-prod \
    --billing-account=XXXXXX-XXXXXX-XXXXXX
```

### 1.3 Enable required APIs
```bash
gcloud services enable compute.googleapis.com
gcloud services enable iam.googleapis.com
```

---

## 2. Provision the Compute Engine VM

### 2.1 Recommended sizing

| Resource | Minimum | Recommended |
|---|---|---|
| Machine type | `e2-medium` (2 vCPU, 4 GB RAM) | `e2-standard-2` (2 vCPU, 8 GB RAM) |
| Boot disk    | 30 GB Standard PD | 50 GB Balanced PD |
| OS image     | Debian 12 / Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| Region/zone  | Closest to your users (e.g. `europe-west1-b`) | — |

> Embedding workloads are CPU- and memory-intensive. On `e2-medium` (4 GB RAM) a swap file is mandatory (see §4.2).

### 2.2 Create the VM
```bash
gcloud compute instances create eigenmind-vm \
    --zone=europe-west1-b \
    --machine-type=e2-standard-2 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-balanced \
    --tags=eigenmind-app
```

### 2.3 Reserve a static external IP (recommended)
A static IP prevents the public address from changing after a reboot — essential if you plan to attach a DNS record.

```bash
gcloud compute addresses create eigenmind-ip \
    --region=europe-west1

# Note the address that is returned
gcloud compute addresses describe eigenmind-ip --region=europe-west1

# Attach it to the VM
gcloud compute instances delete-access-config eigenmind-vm \
    --zone=europe-west1-b \
    --access-config-name="external-nat"

gcloud compute instances add-access-config eigenmind-vm \
    --zone=europe-west1-b \
    --access-config-name="external-nat" \
    --address=<RESERVED_IP>
```

### 2.4 Firewall rules
Open the Streamlit port (`8501`) to the public, restricted by the network tag set above:

```bash
gcloud compute firewall-rules create allow-eigenmind-8501 \
    --direction=INGRESS \
    --action=ALLOW \
    --rules=tcp:8501 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=eigenmind-app
```

> For production deployments, restrict `--source-ranges` to known CIDR blocks, or place the VM behind an HTTPS load balancer with IAP authentication.

---

## 3. Connect to the VM and Install System Dependencies

### 3.1 SSH into the VM
```bash
gcloud compute ssh eigenmind-vm --zone=europe-west1-b
```

### 3.2 Install Python, Docker, Git and OCR tooling
Run the following block on the VM:

```bash
# Update packages
sudo apt-get update && sudo apt-get upgrade -y

# Python 3.10+, build tooling, virtualenv
sudo apt-get install -y python3 python3-venv python3-pip build-essential git

# Docker Engine + Compose plugin
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Allow current user to run docker without sudo
sudo usermod -aG docker $USER
newgrp docker
```

### 3.3 Clone the repository
```bash
# Pick a location, e.g. ~/eigenmind
mkdir -p ~/eigenmind && cd ~/eigenmind

# Clone (replace with your fork if applicable)
git clone https://github.com/merlin-intelligence/eigenmind.git
cd eigenmind

# Track the official repo as 'upstream' — required by the update script in Part II
git remote add upstream https://github.com/merlin-intelligence/eigenmind.git
```

### 3.4 Python environment & dependencies
```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -e .
# Optional extras
pip install -e ".[ocr,gdrive,sharepoint]"
```

### 3.5 Launch Qdrant (Docker)
```bash
docker compose up -d
docker ps | grep qdrant   # verify
```

### 3.6 Configure secrets
Pick one of the two paths described in the [Architecture & Installation Guide](architecture_and_installation_guide.md#step-4-configure-api-keys-secrets):

```bash
# Option A — .env at the repo root
cp .env.example .env
nano .env

# Option B — Streamlit secrets
mkdir -p .streamlit
nano .streamlit/secrets.toml
```

At minimum set `NEBIUS_API_KEY`. Add SharePoint / Google Drive credentials only if those connectors will be used.

### 3.7 Smoke test
Before configuring the service, verify the app starts manually:
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```
Open `http://<EXTERNAL_IP>:8501` in your browser. Once it loads, stop with `Ctrl+C`.

---

## 4. VM Tuning

### 4.1 Timezone & NTP
```bash
sudo timedatectl set-timezone Europe/Paris
```

### 4.2 Swap configuration (mandatory on 4 GB VMs)
To prevent Out-Of-Memory crashes during document embedding:
```bash
sudo fallocate -l 3.3G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Verify with `free -h` — the `Swap` line should report ~3.3G available.

---

## 5. Persistent Service (systemd)

To ensure the app runs automatically on boot and stays running after you disconnect:

1.  **Customize the Template**:
    Open `scripts/eigenmind.service.template` and replace each placeholder:
    - `User=...` / `Group=...` — the Linux user that owns the repo (e.g. `username`).
    - `WorkingDirectory=...` — the absolute path to the cloned repo (e.g. `/home/username/eigenmind/eigenmind`).
    - Every other `...` in the file (in `ExecStart`, `StandardOutput`, `StandardError`) refers to **the same working directory** and must be replaced with that same absolute path. For example, with `WorkingDirectory=/home/username/eigenmind/eigenmind`:
      ```ini
      ExecStart=/home/username/eigenmind/eigenmind/venv/bin/python -m streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
      StandardOutput=append:/home/username/eigenmind/eigenmind/streamlit.log
      StandardError=append:/home/username/eigenmind/eigenmind/streamlit.log
      ```

2.  **Install the Service**:
    ```bash
    # Copy template to the system directory
    sudo cp scripts/eigenmind.service.template /etc/systemd/system/eigenmind.service

    # Reload and Start
    sudo systemctl daemon-reload
    sudo systemctl enable eigenmind
    sudo systemctl start eigenmind
    ```

3.  **Management**:
    - **Check Status**: `sudo systemctl status eigenmind`
    - **View Logs**: `journalctl -u eigenmind -f`
    - **Restart**: `sudo systemctl restart eigenmind`

---

## 6. Data Persistence & Backup

- **Vector Database**: All data is stored in `./qdrant_storage/`. Ensure this directory is backed up.
- **User Data**: OAuth tokens and user profiles are stored in `./user_data/`.
- **Docker**: The Qdrant container is managed via `docker-compose.yml`. If the database fails to connect, verify the container status:
  ```bash
  docker ps | grep qdrant
  ```

### Recommended GCP-native backup
Snapshot the boot disk daily through a Compute Engine snapshot schedule:
```bash
gcloud compute resource-policies create snapshot-schedule eigenmind-daily \
    --region=europe-west1 \
    --max-retention-days=14 \
    --daily-schedule \
    --start-time=02:00

gcloud compute disks add-resource-policies eigenmind-vm \
    --zone=europe-west1-b \
    --resource-policies=eigenmind-daily
```

---

# Part II — Updates

Once the VM is initialised, keep it in sync with the latest code on the `main` branch of the repository. The procedure described here is **manual** — an operator SSHes into the VM and triggers each update explicitly. Automated continuous deployment is intentionally out of scope for now.

> This procedure relies on the `upstream` remote configured in §3.3. If it is missing, run `git remote add upstream https://github.com/merlin-intelligence/eigenmind.git` first.

## 7. Manual Update

### 7.1 Pull the latest code
SSH into the VM, then:
```bash
cd ~/eigenmind/eigenmind

# Fetch and align with the official main branch
git fetch upstream main
git reset --hard upstream/main
```

> `git reset --hard` discards any local change in the working tree. Make sure any patch you wanted to keep is committed and pushed elsewhere first.

### 7.2 Refresh Python dependencies
Only strictly required when `pyproject.toml` / `requirements.txt` changed, but harmless to run every time:
```bash
~/eigenmind/eigenmind/venv/bin/pip install -e .
```

### 7.3 Restart the service
```bash
sudo systemctl restart eigenmind

# Confirm it came back up
sudo systemctl status eigenmind
journalctl -u eigenmind -n 50 --no-pager
```

### 7.4 Rollback procedure
If the new revision misbehaves, roll back to the previous commit:
```bash
cd ~/eigenmind/eigenmind
git reflog                            # find the previous SHA
git reset --hard <previous_sha>
~/eigenmind/eigenmind/venv/bin/pip install -e .
sudo systemctl restart eigenmind
```

---

## 8. Hardening Checklist (Optional but Recommended)

- [ ] Restrict firewall `--source-ranges` to known IPs, or put the app behind an HTTPS load balancer.
- [ ] Terminate TLS with a managed certificate (GCP Load Balancer + managed SSL cert, or Caddy/Nginx in front of port 8501).
- [ ] Enable [OS Login](https://cloud.google.com/compute/docs/oslogin) so SSH access is managed via IAM.
- [ ] Configure [Cloud Monitoring](https://cloud.google.com/monitoring) agent for CPU / memory / disk alerts.
- [ ] Rotate `NEBIUS_API_KEY` and Streamlit user passwords periodically.

---
© 2026 Merlin Intelligence
