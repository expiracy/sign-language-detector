"""
Script to split datasets_v2 data into training and test sets.
Randomly selects 10% of files from each letter folder to move to test/<letter>.
"""

import os
import random
import shutil
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).parent.parent.parent

DATASETS_V2_DIR = BASE_DIR / "data" / "datasets_v2"
TEST_DIR = BASE_DIR / "data" / "test"

# Percentage of files to move to test set
TEST_SPLIT = 0.10

# Exclusion list - datasets to skip (already processed)
EXCLUDED_DATASETS = [
    "ASL_DATA_Far",
    "ASL_DATA_Isa",
    "ASL_DATA_Test",
    "ASL_DATA_Aaron2",
    "ASL_DATA_George_Dark",
    "ASL_DATA_Rohan"
]

def main():
    print(f"Base directory: {BASE_DIR}")
    print(f"Datasets V2 directory: {DATASETS_V2_DIR}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Test split: {TEST_SPLIT * 100}%")
    print(f"Excluded datasets: {EXCLUDED_DATASETS}")
    print("-" * 50)
    
    # Check if datasets_v2 exists
    if not DATASETS_V2_DIR.exists():
        print(f"Error: {DATASETS_V2_DIR} does not exist!")
        return
    
    # Get all subdirectories in datasets_v2 (e.g., ASL_DATA_Far, ASL_DATA_Isa, etc.)
    # Filter out excluded datasets
    dataset_folders = [d for d in DATASETS_V2_DIR.iterdir() 
                       if d.is_dir() and d.name not in EXCLUDED_DATASETS]
    
    if not dataset_folders:
        print("No subdirectories found in datasets_v2!")
        return
    
    print(f"Found {len(dataset_folders)} dataset folders:")
    for folder in dataset_folders:
        print(f"  - {folder.name}")
    print("-" * 50)
    
    # Track statistics
    total_files_moved = 0
    total_files_remaining = 0
    
    # Process each dataset folder
    for dataset_folder in dataset_folders:
        print(f"\nProcessing: {dataset_folder.name}")
        
        # Get all letter subfolders (e.g., A, B, C, etc.)
        letter_folders = [d for d in dataset_folder.iterdir() if d.is_dir()]
        
        for letter_folder in letter_folders:
            letter = letter_folder.name
            
            # Get all files in this letter folder
            files = [f for f in letter_folder.iterdir() if f.is_file()]
            
            if not files:
                print(f"  {letter}: No files found, skipping...")
                continue
            
            # Calculate number of files for test set (at least 1 if there are files)
            num_test_files = max(1, int(len(files) * TEST_SPLIT))
            
            # Randomly select files for test set
            test_files = random.sample(files, min(num_test_files, len(files)))
            
            # Create test directory for this letter if it doesn't exist
            test_letter_dir = TEST_DIR / letter
            test_letter_dir.mkdir(parents=True, exist_ok=True)
            
            # Move selected files to test directory
            for file_path in test_files:
                # Create unique filename to avoid conflicts from different datasets
                # Format: originalname_datasetname.extension
                new_filename = f"{file_path.stem}_{dataset_folder.name}{file_path.suffix}"
                dest_path = test_letter_dir / new_filename
                
                # Handle potential filename conflicts
                counter = 1
                while dest_path.exists():
                    new_filename = f"{file_path.stem}_{dataset_folder.name}_{counter}{file_path.suffix}"
                    dest_path = test_letter_dir / new_filename
                    counter += 1
                
                shutil.move(str(file_path), str(dest_path))
            
            files_moved = len(test_files)
            files_remaining = len(files) - files_moved
            total_files_moved += files_moved
            total_files_remaining += files_remaining
            
            print(f"  {letter}: {files_moved} files moved to test, {files_remaining} files remaining")
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total files moved to test: {total_files_moved}")
    print(f"Total files remaining in datasets_v2: {total_files_remaining}")
    print(f"Test set percentage: {total_files_moved / (total_files_moved + total_files_remaining) * 100:.2f}%")
    print("\nDone!")

if __name__ == "__main__":
    main()
