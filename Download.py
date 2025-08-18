import requests
import shutil
import sys
import os
import time
import hashlib
from pathlib import Path
from configurations.params import NEXUS_DIR, NEXUS_USERNAME
from configurations.global_data import get_global_data

def calculate_file_hash(file_path, algorithm='md5'):
    """Calculate hash of a file for integrity verification."""
    hash_func = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def download_with_retry(url, dest_path, max_retries=3, retry_delay=5):
    """Download a file with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"🔄 Download attempt {attempt + 1} of {max_retries}")
            response = requests.get(url, stream=True, verify=False, timeout=120)
            
            if response.status_code == 200:
                # Save the file with a temporary name first
                temp_path = f"{dest_path}.tmp"
                with open(temp_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                
                # Move temp file to final destination
                shutil.move(temp_path, dest_path)
                print(f"✅ Download successful on attempt {attempt + 1}")
                return True
            else:
                print(f"❌ Download failed with status code: {response.status_code}")
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Download error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        except Exception as e:
            print(f"❌ Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    return False

def verify_extracted_files():
    """Verify that extracted files are present and valid."""
    expected_files = [
        ("/tmp/skku_slm_resnet50_vd_ocr99.pt.params", 100),  # Minimum size in bytes
        ("/tmp/ssd_512_resnet101_v2_voc-2cc0f93e.params", 100),
        ("/tmp/ssd_512_resnet101_resume_train_four_classes_custom-epoch-26.params", 100),
        ("/tmp/detection_model/efficientdet.onnx", 100),
        ("/tmp/efficientdet_conf.yml", 10),
    ]
    
    expected_dirs = [
        "/tmp/en_core_web_sm-3.8.0/en_core_web_sm/en_core_web_sm-3.8.0/",
        "/tmp/detection_model/"
    ]
    
    # Check directories
    for dir_path in expected_dirs:
        if not os.path.isdir(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            return False
        if not os.access(dir_path, os.R_OK):
            print(f"❌ Directory not readable: {dir_path}")
            return False
    
    # Check files
    for file_path, min_size in expected_files:
        if not os.path.isfile(file_path):
            print(f"❌ Missing file: {file_path}")
            return False
        
        file_size = os.path.getsize(file_path)
        if file_size < min_size:
            print(f"❌ File too small: {file_path} ({file_size} bytes)")
            return False
        
        if not os.access(file_path, os.R_OK):
            print(f"❌ File not readable: {file_path}")
            return False
    
    print("✅ All extracted files verified successfully")
    return True

def safe_extract(archive_path, extract_to="/tmp"):
    """Safely extract archive with error handling."""
    try:
        # Create a temporary extraction directory
        temp_extract_dir = f"{extract_to}/extract_temp_{os.getpid()}"
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        # Extract to temporary directory first
        shutil.unpack_archive(archive_path, temp_extract_dir)
        print(f"📦 Extracted to temporary directory: {temp_extract_dir}")
        
        # Move files from temp directory to final destination
        for item in os.listdir(temp_extract_dir):
            source = os.path.join(temp_extract_dir, item)
            destination = os.path.join(extract_to, item)
            
            # Remove destination if it exists
            if os.path.exists(destination):
                if os.path.isdir(destination):
                    shutil.rmtree(destination)
                else:
                    os.remove(destination)
            
            # Move from temp to final location
            shutil.move(source, destination)
            print(f"📁 Moved: {item} to {extract_to}")
        
        # Clean up temp directory
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        return True
        
    except shutil.ReadError as e:
        print(f"❌ Error unpacking archive: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during extraction: {e}")
        return False

def clean_partial_downloads():
    """Clean up any partial or temporary download files."""
    patterns = [
        "/tmp/*.tmp",
        "/tmp/models.zip",
        "/tmp/extract_temp_*"
    ]
    
    import glob
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path, ignore_errors=True)
                else:
                    os.remove(file_path)
                print(f"🗑️ Cleaned up: {file_path}")
            except:
                pass

def download_models_with_lock(SA_keepie_key):
    """Download models with proper error handling and verification."""
    
    # Clean up any partial downloads first
    clean_partial_downloads()
    
    password = get_global_data(SA_keepie_key)
    if not password:
        print(f"❌ Failed to get password for key: {SA_keepie_key}")
        return False
    
    url1 = f"https://{NEXUS_USERNAME}:{password}@nexus305.systems.uk.hsbc:8081/nexus/repository/maven-hsbc-gbm/com/hsbc/gbm/dcrest/992994/{NEXUS_DIR}/1.1/{NEXUS_DIR}-1.1.zip"
    print(f"📍 Download URL prepared (credentials hidden)")
    
    temp_zip_path = "/tmp/models.zip"
    
    try:
        # Download with retry
        print("🔄 Starting model download...")
        if not download_with_retry(url1, temp_zip_path):
            print("❌ Failed to download models after all retries")
            return False
        
        # Verify download
        if not os.path.exists(temp_zip_path):
            print("❌ Downloaded file not found")
            return False
        
        file_size = os.path.getsize(temp_zip_path)
        print(f"📊 Downloaded file size: {file_size:,} bytes")
        
        if file_size < 1000:  # Minimum expected size
            print("❌ Downloaded file is too small, likely corrupted")
            return False
        
        # Extract archive
        print("📦 Extracting models...")
        if not safe_extract(temp_zip_path, "/tmp"):
            print("❌ Failed to extract models")
            return False
        
        # Verify extraction
        if not verify_extracted_files():
            print("❌ Extracted files verification failed")
            return False
        
        print("✅ Model download and extraction completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error in download_models: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up zip file
        if os.path.exists(temp_zip_path):
            try:
                os.remove(temp_zip_path)
                print("🗑️ Cleaned up download archive")
            except:
                pass

def download_models(SA_keepie_key):
    """Legacy function for backward compatibility."""
    return download_models_with_lock(SA_keepie_key)
