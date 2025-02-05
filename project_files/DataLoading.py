import torch
from sympy.solvers.diophantine.diophantine import reconstruct
from torch.utils.data import Dataset, Sampler, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2



def collate_fn(batch):
    """Ensures the batch is a list of dictionaries, not a single merged dictionary."""
    return batch  # Returns the list of dictionaries as is

def estimate_memory_usage(num_prompts):
    """Estimate memory usage based on prompt count."""
    base_memory = 50  # MB per tile
    per_prompt_memory = 10  # MB per prompt
    return base_memory + per_prompt_memory * num_prompts

def add_buffer_to_tile(tile, original_image, buffer):
    """
    Add buffer to a tile. For edges, add zeros as padding.

    Args:
        tile (dict): A dictionary containing tile data with keys "tile_index", "coords" (x, y), "num_prompts", and "tile_data".
        original_image (numpy array): The original image array.
        buffer (int): The buffer size to add around the tile.

    Returns:
        dict: The tile with added buffer.
    """
    tile_index = tile["tile_index"]
    x, y = tile["coords"]
    tile_data = tile["tile_data"]

    height, width, channels = original_image.shape
    tile_height, tile_width, _ = tile_data.shape

    # Create the buffer-padded tile with zeros
    padded_tile = np.zeros((tile_height + 2 * buffer, tile_width + 2 * buffer, channels), dtype=np.uint8)

    # Copy the tile data into the padded tile
    padded_tile[buffer:buffer + tile_height, buffer:buffer + tile_width] = tile_data

    # Add top buffer
    if y - buffer >= 0:
        padded_tile[:buffer, buffer:buffer + tile_width] = original_image[y - buffer:y, x:x + tile_width]
    else:
        padded_tile[:buffer, buffer:buffer + tile_width] = np.zeros((buffer, tile_width, channels))

    # Add bottom buffer
    if y + tile_height + buffer <= height:
        padded_tile[buffer + tile_height:, buffer:buffer + tile_width] = original_image[
            y + tile_height:y + tile_height + buffer, x:x + tile_width]
    else:
        padded_tile[buffer + tile_height:, buffer:buffer + tile_width] = np.zeros((buffer, tile_width, channels))

    # Add left buffer
    if x - buffer >= 0:
        padded_tile[buffer:buffer + tile_height, :buffer] = original_image[y:y + tile_height, x - buffer:x]
    else:
        padded_tile[buffer:buffer + tile_height, :buffer] = np.zeros((tile_height, buffer, channels))

    # Add right buffer
    if x + tile_width + buffer <= width:
        padded_tile[buffer:buffer + tile_height, buffer + tile_width:] = original_image[
            y:y + tile_height, x + tile_width:x + tile_width + buffer]
    else:
        padded_tile[buffer:buffer + tile_height, buffer + tile_width:] = np.zeros((tile_height, buffer, channels))

    # Add top-left diagonal padding
    if y - buffer >= 0 and x - buffer >= 0:
        padded_tile[:buffer, :buffer] = original_image[y - buffer:y, x - buffer:x]
    else:
        padded_tile[:buffer, :buffer] = np.zeros((buffer, buffer, channels))

    # Add top-right diagonal padding
    if y - buffer >= 0 and x + tile_width + buffer <= width:
        padded_tile[:buffer, buffer + tile_width:] = original_image[y - buffer:y, x + tile_width:x + tile_width + buffer]
    else:
        padded_tile[:buffer, buffer + tile_width:] = np.zeros((buffer, buffer, channels))

    # Add bottom-left diagonal padding
    if y + tile_height + buffer <= height and x - buffer >= 0:
        padded_tile[buffer + tile_height:, :buffer] = original_image[y + tile_height:y + tile_height + buffer, x - buffer:x]
    else:
        padded_tile[buffer + tile_height:, :buffer] = np.zeros((buffer, buffer, channels))

    # Add bottom-right diagonal padding
    if y + tile_height + buffer <= height and x + tile_width + buffer <= width:
        padded_tile[buffer + tile_height:, buffer + tile_width:] = original_image[
            y + tile_height:y + tile_height + buffer, x + tile_width:x + tile_width + buffer]
    else:
        padded_tile[buffer + tile_height:, buffer + tile_width:] = np.zeros((buffer, buffer, channels))

    return {
        "tile_index": tile["tile_index"],
        "coords": tile["coords"],
        "num_prompts": tile["num_prompts"],
        "tile_data": padded_tile
    }

def remove_buffer_from_tile(tile, buffer):
    """
    Remove buffer from a tile.

    Args:
        tile (dict): A dictionary containing tile data with keys "tile_index", "coords" (x, y), "num_prompts", and "tile_data".
        buffer (int): The buffer size to remove from around the tile.

    Returns:
        dict: The tile with the buffer removed.
    """
    tile_index = tile["tile_index"]
    x, y = tile["coords"]
    tile_data = tile["tile_data"]

    # Remove the buffer from each side
    tile_height, tile_width, _ = tile_data.shape
    original_tile_data = tile_data[buffer:tile_height-buffer, buffer:tile_width-buffer, :]

    return {
        "tile_index": tile_index,
        "coords": (x, y),
        "num_prompts": tile["num_prompts"],
        "tile_data": original_tile_data
    }

def reassemble_image(dataset, original_image_shape, buffer):
    """
    Reassemble tiles from an ImageTileDataset back into the full image.

    Args:
        dataset (ImageTileDataset): The dataset containing tiles.
        original_image_shape (tuple): The shape of the original image (height, width, channels).
        tile_size (int): The size of each tile (excluding buffer).

    Returns:
        numpy array: The reassembled full image.
    """
    if not isinstance(dataset, ImageTileDataset):
        raise TypeError("Input must be an instance of ImageTileDataset.")

    height, width, channels = original_image_shape

    # Create an empty image array with the original image dimensions
    full_image = np.zeros((height, width, channels), dtype=np.uint8)

    for tile in dataset:
        og_tile = remove_buffer_from_tile(tile, buffer)
        x, y = og_tile["coords"]
        tile_data = og_tile["tile_data"]
        tile_height, tile_width, _ = tile_data.shape

        y_end = y + tile_height
        x_end = x + tile_width

        full_image[y:y_end, x:x_end, :] = tile_data[:y_end - y, :x_end - x, :]

    return full_image

def calculate_mse(image1, image2):
    """
    Calculate Mean Squared Error (MSE) between two images.

    Args:
        image1 (numpy array): The first image.
        image2 (numpy array): The second image.

    Returns:
        float: The Mean Squared Error between the two images.
    """
    return np.mean((image1 - image2) ** 2)

class ImageTileDataset(Dataset):
    def __init__(self, image=None, tile_list=None, centroids=None, tile_size=None, buffer=None):
        """
        Args:
            image (numpy array, optional): The original large image of shape (10980, 10980, 3).
            tile_list (list, optional): A list of tile_data.
            tile_size (int, optional): The size of each tile.
            buffer (int, optional): The buffer size to add around the tile.
        """
        self.tiles = []

        if image is not None:
            self._initialize_from_image(image, centroids ,tile_size, buffer)
        elif tile_list is not None:
            self._initialize_from_tiles(tile_list)

    def _initialize_from_image(self, image, centroids, tile_size, buffer):
        tile_index = 0
        height, width, channels = image.shape

        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Handle tiles at the edges
                y_end = min(height, y + tile_size)
                x_end = min(width, x + tile_size)
                tile_data = image[y:y_end, x:x_end, :]
                tile_centroids = centroids[y:y_end, x:x_end]
                num_prompts = np.count_nonzero(tile_centroids)

                tile = {
                    "tile_index": tile_index,
                    "coords": (x, y),
                    "num_prompts": num_prompts,
                    "tile_data": tile_data
                }
                padded_tile = add_buffer_to_tile(tile, image, buffer)
                self.tiles.append(padded_tile)
                tile_index += 1

    def _initialize_from_tiles(self, tile_list):
        self.tiles = tile_list

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        return self.tiles[index]

class MemoryAwareSampler(Sampler):
    def __init__(self, dataset, gpu_memory_limit):
        """
        Args:
            dataset (ImageTileDataset): The dataset containing tiles.
            gpu_memory_limit (int): Approximate max GPU memory in MB.
        """
        self.dataset = dataset
        self.gpu_memory_limit = gpu_memory_limit

        # Precompute memory estimates
        self.memory_usage = [self.estimate_memory_usage(tile["num_prompts"]) for tile in dataset]

        # Sort tiles by memory usage (descending order)
        self.sorted_indices = np.argsort(-np.array(self.memory_usage))

    def estimate_memory_usage(self, num_prompts):
        """Estimate memory usage based on prompt count."""
        base_memory = 50  # MB per tile
        per_prompt_memory = 10  # MB per prompt
        return base_memory + per_prompt_memory * num_prompts

    def __iter__(self):
        batch = []
        current_memory = 0

        for idx in self.sorted_indices:
            tile = self.dataset[idx]
            tile_memory = self.memory_usage[idx]

            if current_memory + tile_memory > self.gpu_memory_limit:
                yield batch  # Return batch when full
                batch = []
                current_memory = 0

            batch.append(idx)
            current_memory += tile_memory

        if batch:
            yield batch  # Return any remaining tiles

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    before_img = cv2.imread('data/before.png')
    before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
    after_img = cv2.imread('data/after.png')
    after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
    diff_mask = cv2.imread('data/mask.png')

    tile_size = 70
    buffer = 30
    gpu_memory_limit = 4000  # Assume 400MB available for batching

    # produce centroids and coords from mask - complete this
    centroids = np.zeros_like(diff_mask)
    # Randomly choose 1000 unique positions in the array
    indices = np.random.choice(centroids.size, 1000, replace=False)
    # Set the chosen positions to 255
    centroids.flat[indices] = 255

    before_dataset = ImageTileDataset(image=before_img, centroids=centroids, tile_size=tile_size, buffer=buffer)
    before_sampler = MemoryAwareSampler(before_dataset, gpu_memory_limit // 2)
    before_dataloader = DataLoader(before_dataset, batch_sampler=before_sampler, num_workers=0, collate_fn=collate_fn)
    after_dataset = ImageTileDataset(image=after_img, centroids=centroids, tile_size=tile_size, buffer=buffer)
    after_sampler = MemoryAwareSampler(after_dataset, gpu_memory_limit // 2)
    after_dataloader = DataLoader(after_dataset, batch_sampler=after_sampler, num_workers=0, collate_fn=collate_fn)
    mask_dataset = ImageTileDataset(image=diff_mask, centroids=centroids, tile_size=tile_size, buffer=buffer)
    mask_sampler = MemoryAwareSampler(mask_dataset, gpu_memory_limit // 2) # no prompts - no real memory limit
    mask_dataloader = DataLoader(mask_dataset, batch_sampler=mask_sampler, num_workers=0, collate_fn=collate_fn)

    result_tiles = []
    for i, (before_batch, after_batch, mask_batch) in enumerate(
            zip(before_dataloader, after_dataloader, mask_dataloader)):
        memory = 0
        for item in before_batch:
            memory += estimate_memory_usage(item["num_prompts"])
        memory = memory * 2 # account for after batch as well
        print("Processing batch:", i, "using", memory, "memory")

        # use segment_batch(before_batch, after_batch, mask_batch) to get result_batch
        result_batch = mask_batch
        for item in result_batch:
            result_tiles.append(item)
    reconstructed_mask_dataset = ImageTileDataset(tile_list=result_tiles)

    # Reassemble the full mask from the tiles
    reconstructed_mask = reassemble_image(reconstructed_mask_dataset, diff_mask.shape, buffer)
