import hydra
import argparse
from omegaconf import OmegaConf
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
from tqdm import tqdm
import threading
import random

def load_batch_items(dataset, start_idx, batch_size):
    """Load a batch of consecutive items from the dataset"""
    results = []
    for i in range(start_idx, min(start_idx + batch_size, len(dataset))):
        _ = dataset[i]  # Trigger __getitem__ to activate caching
        results.append((i, True))
    return results

def cache_dataset(cfg: dict, num_workers: int = 8, shuffle: bool = False, batch_size: int = 32):
    """
    Cache the dataset by loading all items using custom multi-threading.
    This ensures that all data is loaded and cached without returning data.
    
    Args:
        cfg: Dataset configuration
        num_workers: Number of worker threads
        shuffle: Whether to shuffle the dataset indices before processing
        batch_size: Number of consecutive items each worker processes (default: 32)
    """
    print("Starting dataset caching...")
    print(OmegaConf.to_yaml(cfg))

    # Instantiate the dataset
    dataset = hydra.utils.instantiate(cfg)
    dataset_size = len(dataset)
    print(f"Dataset instantiated with {dataset_size} items")
    print(f"Using batch size of {batch_size} items per worker")

    dataset.cache_all_pcd(
        batch_size=batch_size,
        num_reader_threads=num_workers // 2,
        num_processor_threads=num_workers - (num_workers // 2),
        max_queue_size=batch_size * (num_workers // 2) * 8,
        randperm=shuffle
    )
    
    # Progress tracking
    progress_bar = tqdm(total=dataset_size, desc="Caching dataset")
    lock = threading.Lock()
    
    def update_progress(num_items):
        with lock:
            progress_bar.update(num_items)
    
    # Use ThreadPoolExecutor for parallel loading
    print(f"Starting dataset iteration with {num_workers} workers...")
    success_count = 0
    failed_indices = []
    
    # Create batches of consecutive indices
    batches = []
    for start_idx in range(0, dataset_size, batch_size):
        batches.append(start_idx)
    
    if shuffle:
        random.shuffle(batches)
        print(f"Batch start indices shuffled for random processing order")
    
    print(f"Created {len(batches)} batches of size {batch_size}")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit batch tasks
        futures = {
            executor.submit(load_batch_items, dataset, start_idx, batch_size): start_idx 
            for start_idx in batches
        }
        
        # Collect results
        for future in as_completed(futures):
            batch_results = future.result()
            batch_success = 0
            
            for idx, success in batch_results:
                if success:
                    success_count += 1
                    batch_success += 1
                else:
                    failed_indices.append(idx)
            
            # Update progress with the number of items processed in this batch
            update_progress(len(batch_results))
    
    progress_bar.close()
    
    print(f"Dataset caching completed!")
    print(f"Successfully processed: {success_count}/{dataset_size} items")
    if failed_indices:
        print(f"Failed indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
        print(f"Total failed: {len(failed_indices)}")
    else:
        print("All items processed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache dataset by loading all items")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle batch order for random processing order")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of consecutive items each worker processes")
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Apply command line overrides
    cfg['use_cached_metadata'] = True
    
    print(f"Using {args.num_workers} worker threads")
    print(f"Using batch size of {args.batch_size} items per worker")
    if args.shuffle:
        print("Shuffle mode enabled - batches will be processed in random order")
    
    # Cache the dataset
    try:
        cache_dataset(cfg, num_workers=args.num_workers, shuffle=args.shuffle, batch_size=args.batch_size)
    except KeyboardInterrupt:
        print("\nCaching interrupted by user")
    except Exception as e:
        print(f"Error during caching: {e}")
        raise
