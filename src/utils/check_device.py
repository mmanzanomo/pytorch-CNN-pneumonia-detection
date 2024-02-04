import torch


def set_device():
    """
    Set and return the appropriate PyTorch device based on GPU availability.

    Returns:
        torch.device: PyTorch device, either 'cuda' if GPU is available or 'cpu' if not.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
