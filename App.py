# import pvt
import sys
import uvicorn
import urllib3
import platform
import subprocess
import os
import time
import fcntl
import hashlib
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from configurations.global_data import (set_global_data, get_all_global_data)
from utils import keepie_utils
from configurations.params import *
from model_utils.download_utils import download_models_with_lock
from Auth.auth import secure
from routers.process_documents import visual_dcrest
from source.detection.model_load import load_model, get_model

# Suppress InsecureRequestsWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(root_path="/dcrest-visual-extraction-service")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
uname = platform.uname()

# Define lock and flag file paths
LOCK_FILE = "/tmp/model_download.lock"
DOWNLOAD_COMPLETE_FLAG = "/tmp/model_download.complete"
MODEL_LOADED_FLAG = "/tmp/model_loaded.complete"
MAX_WAIT_TIME = 300  # Maximum wait time in seconds
CHECK_INTERVAL = 2  # Check interval in seconds

def get_uptime_millis():
    with open("/proc/uptime", "r") as f:
        uptime_seconds = float(f.readline().split()[0])
    return int(uptime_seconds * 1000)

def wait_for_file(file_path, timeout=MAX_WAIT_TIME):
    """Wait for a file to exist with timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            return True
        time.sleep(CHECK_INTERVAL)
    return False

def acquire_lock(lock_file_path, timeout=30):
    """Acquire a file lock with timeout."""
    lock_file = open(lock_file_path, 'w')
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_file
        except IOError:
            time.sleep(0.1)
    
    raise TimeoutError(f"Could not acquire lock on {lock_file_path} within {timeout} seconds")

def release_lock(lock_file):
    """Release a file lock."""
    if lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()

def verify_model_files():
    """Verify that all required model files are present and accessible."""
    required_files = [
        "/tmp/skku_slm_resnet50_vd_ocr99.pt.params",
        "/tmp/ssd_512_resnet101_v2_voc-2cc0f93e.params",
        "/tmp/ssd_512_resnet101_resume_train_four_classes_custom-epoch-26.params",
        "/tmp/detection_model/efficientdet.onnx",
        "/tmp/efficientdet_conf.yml",
        "/tmp/en_core_web_sm-3.8.0/en_core_web_sm/en_core_web_sm-3.8.0/"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Missing required file: {file_path}")
            return False
        
        # Check if file/directory is accessible
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'rb') as f:
                    # Try to read first few bytes to ensure file is accessible
                    f.read(1)
            except Exception as e:
                print(f"Cannot access file {file_path}: {e}")
                return False
    
    return True

def get_worker_id():
    """Generate a unique worker ID based on process ID and hostname."""
    hostname = platform.node()
    pid = os.getpid()
    return f"{hostname}_{pid}"

uptime_millis = get_uptime_millis()
version = os.getenv("APP_VERSION")
WORKER_ID = get_worker_id()

@app.get("/health", status_code=200, description="Status report of api and system")
def health():
    return {
        "healthy": "true",
        "efimid": "9929948",
        "server": uname.node,
        "componentName": "dcrest-visual-extraction-service",
        "version": version,
        "description": "Description of this service",
        "sourceCodeRepoUrl": "https://alm-github.wellsfargo.com/abcd/dcrest-visual-extraction-service",
        "documentationUrl": "N/A",
        "apiSpecificationUrl": "https://dcrest-visual-extraction-service-url/openapi.json",
        "businessImpact": "This service has business impact",
        "runtime": {
            "name": "PYTHON",
            "version": sys.version
        },
        "uptimeInMillis": uptime_millis,
        "workerId": WORKER_ID
    }

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(req: Request):
    root_path = req.scope.get("root_path", "").rstrip("/")
    openapi_url = root_path + app.openapi_url
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title="visual_detection",
    )

@app.get("/ready", status_code=200, description="Status report of api and system")
def ready():
    model_ready = False
    try:
        model = get_model()
        if "session" in model and model["session"] is not None:
            model_ready = True
    except:
        pass
    
    return {
        "server": uname.node,
        "service_name": "dcrest-visual-extraction-service",
        "status": "alive",
        "model_ready": model_ready,
        "workerId": WORKER_ID
    }

@app.post("/secret-receipt", status_code=200, description="Getting the secret keys")
async def secret_receipt(secretName: str = Form(), secretValue: str = Form(), Content_Type: str = Header()):
    print(f"[Worker {WORKER_ID}] secretName : {secretName}")
    
    """
    Write data to config with proper synchronization
    """
    if secretName and secretValue:
        if secretName == "dcrestai-gb-dcrest-svc-aco-doctager-pwd":
            set_global_data("dcrestai-gb-dcrest-svc-aco-doctager-pwd", secretValue)
            print(f"[Worker {WORKER_ID}] Received password for dcrestai-gb-dcrest-svc-aco-doctager-pwd")
            
            # Check if download is already complete
            if os.path.exists(DOWNLOAD_COMPLETE_FLAG):
                print(f"[Worker {WORKER_ID}] Models already downloaded by another worker.")
            else:
                lock_file = None
                try:
                    # Try to acquire lock for downloading
                    print(f"[Worker {WORKER_ID}] Attempting to acquire download lock...")
                    lock_file = acquire_lock(LOCK_FILE, timeout=5)
                    
                    # Double-check if download is complete (another worker might have finished)
                    if os.path.exists(DOWNLOAD_COMPLETE_FLAG):
                        print(f"[Worker {WORKER_ID}] Models already downloaded by another worker (after acquiring lock).")
                    else:
                        print(f"[Worker {WORKER_ID}] Acquired lock, starting model download...")
                        
                        # Download models
                        success = download_models_with_lock("dcrestai-gb-dcrest-svc-aco-doctager-pwd")
                        
                        if success:
                            # Verify all files are present
                            if verify_model_files():
                                # Create completion flag
                                Path(DOWNLOAD_COMPLETE_FLAG).touch()
                                print(f"[Worker {WORKER_ID}] Model download completed successfully.")
                            else:
                                print(f"[Worker {WORKER_ID}] Model download failed - not all files present.")
                                raise Exception("Model download verification failed")
                        else:
                            print(f"[Worker {WORKER_ID}] Model download failed.")
                            raise Exception("Model download failed")
                
                except TimeoutError:
                    # Another worker is downloading, wait for completion
                    print(f"[Worker {WORKER_ID}] Another worker is downloading, waiting for completion...")
                    if wait_for_file(DOWNLOAD_COMPLETE_FLAG, timeout=MAX_WAIT_TIME):
                        print(f"[Worker {WORKER_ID}] Download completed by another worker.")
                    else:
                        print(f"[Worker {WORKER_ID}] Timeout waiting for model download.")
                        raise Exception("Timeout waiting for model download")
                
                finally:
                    if lock_file:
                        release_lock(lock_file)
        
        # Check if both secrets are available and models are downloaded
        if "dcrestai-gb-dcrest-svc-aco-pwd" in get_all_global_data().keys():
            if "dcrestai-gb-svceco-doctage-pwd" in get_all_global_data().keys():
                
                # Wait for download to complete
                if not os.path.exists(DOWNLOAD_COMPLETE_FLAG):
                    print(f"[Worker {WORKER_ID}] Waiting for model download to complete...")
                    if not wait_for_file(DOWNLOAD_COMPLETE_FLAG, timeout=MAX_WAIT_TIME):
                        print(f"[Worker {WORKER_ID}] Timeout waiting for model download.")
                        return {"error": "Model download timeout"}
                
                # Verify files before loading
                if not verify_model_files():
                    print(f"[Worker {WORKER_ID}] Model files verification failed.")
                    return {"error": "Model files not accessible"}
                
                # Check if model is already loaded
                model = get_model()
                if "session" not in model or model["session"] is None:
                    # Try to load model with lock
                    load_lock_file = None
                    try:
                        load_lock_file = acquire_lock("/tmp/model_load.lock", timeout=30)
                        
                        # Double-check if model is loaded
                        model = get_model()
                        if "session" not in model or model["session"] is None:
                            print(f"[Worker {WORKER_ID}] Loading model...")
                            load_model()
                            print(f"[Worker {WORKER_ID}] Model loaded successfully.")
                        else:
                            print(f"[Worker {WORKER_ID}] Model already loaded by another worker.")
                    
                    except Exception as e:
                        print(f"[Worker {WORKER_ID}] Error loading model: {e}")
                        raise
                    
                    finally:
                        if load_lock_file:
                            release_lock(load_lock_file)
                else:
                    print(f"[Worker {WORKER_ID}] Model already loaded.")
    
    # return the secret
    return {"secretName": secretName, "status": "received", "workerId": WORKER_ID}

# Startup event to clean up stale lock files
@app.on_event("startup")
async def startup_event():
    """Clean up stale lock files on startup."""
    print(f"[Worker {WORKER_ID}] Starting up...")
    
    # Only clean up if this is the first worker (based on some condition)
    # You might want to implement a more sophisticated check here
    if not os.path.exists("/tmp/startup_complete.flag"):
        try:
            # Clean up old lock files
            for file_path in [LOCK_FILE, "/tmp/model_load.lock"]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"[Worker {WORKER_ID}] Cleaned up stale lock file: {file_path}")
                    except:
                        pass
            
            # Create startup flag
            Path("/tmp/startup_complete.flag").touch()
        except:
            pass

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    print(f"[Worker {WORKER_ID}] Shutting down...")

# Routers
app.include_router(secure)
app.include_router(visual_dcrest)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    print(f"Starting with {workers} workers")
    uvicorn.run(
        "Application:app",
        host=host,
        port=port,
        log_level="debug",
        workers=workers,
    )
