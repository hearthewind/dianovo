import torch
import torch.nn as nn
import torch.nn.functional as F

def noble_loss(transformer_embedding, contrastive_label, reduction='mean'):
    """
    Calculate the consistency loss based on the given contrastive_label.

    Parameters:
    transformer_embedding: (batch_size, seq_len, embedding_dim) tensor of transformer embeddings
    contrastive_label: (batch_size, seq_len) tensor of type labels for each element in the sequence
    reduction: 'mean', 'sum', or 'none' indicating the reduction method over the loss

    Returns:
    The calculated consistency loss.
    """
    # Calculate all pair combinations of embeddings and labels within each batch
    emb_pairs = transformer_embedding.unsqueeze(2) - transformer_embedding.unsqueeze(1)
    label_pairs = contrastive_label.unsqueeze(2) == contrastive_label.unsqueeze(1)

    # Calculate pairwise distance for all pairs
    pairwise_distances = torch.norm(emb_pairs, p=2, dim=3)

    # Calculate loss for positive pairs (same label)
    positive_loss = torch.where(label_pairs, torch.min(pairwise_distances, torch.ones_like(pairwise_distances)),
                                torch.zeros_like(pairwise_distances)) ** 2

    # Calculate loss for negative pairs (different label)
    max_part = torch.relu(1 - pairwise_distances)
    negative_loss = torch.where(~label_pairs, torch.max(max_part, torch.zeros_like(max_part)),
                                torch.zeros_like(pairwise_distances)) ** 2

    # Combine losses
    combined_loss = positive_loss + negative_loss

    # Reduce the loss as required
    if reduction == 'mean':
        # We divide by seq_len**2 because we have seq_len**2 pairs per example in the batch
        loss = combined_loss.sum() / (transformer_embedding.size(0) * transformer_embedding.size(1) ** 2)
    elif reduction == 'sum':
        loss = combined_loss.sum()
    elif reduction == 'none':
        loss = combined_loss
    else:
        raise ValueError("Reduction method not recognized. Use 'mean', 'sum', or 'none'.")

    return loss


def l2_normalize(v, axis=-1):
    norm = v.norm(p=2, dim=axis, keepdim=True)
    return v / (norm + 1e-10)

class ClipLoss(nn.Module):
    def __init__(self, temperature_initial=0.07, temperature_max=100.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature_initial))
        self.temperature_max = temperature_max

    def forward(self, encoded_tokens, contrastive_label, reduction='mean'):
        # Normalize the token embeddings
        I_e = l2_normalize(encoded_tokens, axis=-1)

        # Compute scaled pairwise cosine similarities
        # batch_size, num_tokens, _ = I_e.shape
        logits = torch.bmm(I_e, I_e.transpose(1, 2)) / self.clamped_temperature()

        # Create the binary target similarity matrix
        labels = contrastive_label.unsqueeze(2) == contrastive_label.unsqueeze(1)
        labels = labels.float()

        # Compute binary cross-entropy loss for all pairs
        loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

        # Apply reduction method
        if reduction == 'mean':
            loss = loss_matrix.mean()
        elif reduction == 'sum':
            loss = loss_matrix.sum()
        elif reduction == 'none':
            loss = loss_matrix
        else:
            raise ValueError("Invalid reduction method. Use 'mean', 'sum', or 'none'.")

        return loss

    def clamped_temperature(self):
        # Clamp the temperature parameter to avoid it becoming too large
        return torch.clamp(self.temperature, max=self.temperature_max)

def infonce_loss(transformer_embedding, contrastive_label, reduction='mean', temperature=0.1):
    batch_size, seq_len, embedding_dim = transformer_embedding.shape

    # Normalize embeddings to unit length
    embedding_norm = F.normalize(transformer_embedding, p=2, dim=-1)

    # Compute cosine similarity matrix
    logits = torch.bmm(embedding_norm, embedding_norm.transpose(1, 2)) / temperature

    # Create a mask for self-comparisons
    mask = torch.eye(seq_len, dtype=torch.bool, device=transformer_embedding.device)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # Expand mask for all batches

    # Apply mask to logits: set self-comparisons to '-inf'
    logits = logits.masked_fill(mask, float('-inf'))

    # Construct labels for positive samples
    labels = contrastive_label.unsqueeze(2) == contrastive_label.unsqueeze(1)
    positives_mask = labels & ~torch.eye(seq_len, dtype=torch.bool, device=logits.device).unsqueeze(0)

    # Calculate log probabilities
    log_probs = F.log_softmax(logits, dim=2)

    # Compute InfoNCE loss
    loss_elements = -log_probs[positives_mask]

    if reduction == 'mean':
        loss = loss_elements.mean()
    elif reduction == 'sum':
        loss = loss_elements.sum()
    elif reduction == 'none':
        loss = loss_elements
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Use 'mean', 'sum', or 'none'.")

    return loss