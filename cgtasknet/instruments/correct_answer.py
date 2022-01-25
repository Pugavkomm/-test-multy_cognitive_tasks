import torch


def _is_correct_output(real_output: torch.Tensor, target_output: torch.Tensor) -> bool:
    # compute mean:
    if len(real_output.shape) == 3:
        real_output = real_output.reshape(real_output.shape[0], real_output.shape[2])
    if len(target_output.shape) == 3:
        target_output = target_output.reshape(
            target_output.shape[0], target_output.shape[2]
        )
    target_mean = torch.mean(target_output, axis=0)
    real_mean = torch.mean(real_output, axis=0)
    return torch.argmax(target_mean).item() == torch.argmax(real_mean).item()


def _is_correct_output_batches(
    real_output: torch.tensor, target_output: torch.tensor
) -> torch.tensor:
    target_mean = torch.mean(target_output, axis=0)
    real_mean = torch.mean(real_output, axis=0)
    result = torch.zeros(real_output.shape[1], dtype=bool)
    for i in range(real_mean.shape[0]):
        result[i] = torch.argmax(target_mean[i]) == torch.argmax(real_mean[i])
    return result
