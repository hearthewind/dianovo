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


def corrected_simclr_loss(transformer_embedding, contrastive_label, reduction='mean', temperature=0.07, epsilon=1e-8):
    batch_size, seq_len, _ = transformer_embedding.size()
    total_loss = []

    for i in range(batch_size):
        # Extract embeddings and labels for the current sample in the batch.
        emb = transformer_embedding[i]  # Shape: (seq_len, embedding_dim)
        labels = contrastive_label[i]  # Shape: (seq_len,)

        # Compute normalized similarity matrix for the embeddings.
        norm_emb = F.normalize(emb, p=2, dim=1)
        sim_matrix = torch.mm(norm_emb, norm_emb.t()) / temperature  # Shape: (seq_len, seq_len)

        # Create a mask for identifying positive pairs, excluding self-comparisons.
        mask_positive = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float() - torch.eye(seq_len, device=emb.device)

        # Compute the exponential similarity for all pairs and mask out self-similarities.
        exp_sim = torch.exp(sim_matrix) * (1 - torch.eye(seq_len, device=emb.device))

        # Compute sum of exponentials for all pairs (including positives), used in the denominator.
        sum_exp_all = torch.sum(exp_sim, dim=1, keepdim=True)  # Shape: (seq_len, 1)

        # Compute the contribution of positives only, by using the mask.
        sum_exp_positives = torch.sum(exp_sim * mask_positive, dim=1, keepdim=True)  # Shape: (seq_len, 1)

        # Compute the contrastive loss for each element using the masked positive values.
        # The loss is computed by considering the sum of all positive similarities for each anchor against the sum of all similarities (excluding self).
        loss_per_element = -torch.log((sum_exp_positives + epsilon) / (sum_exp_all + epsilon))

        # Store the computed loss for this batch element.
        total_loss.append(loss_per_element)

    # Stack to get the batch-wise loss and apply reduction.
    total_loss = torch.cat(total_loss, dim=0)  # Shape: (batch_size * seq_len, 1)
    if reduction == 'mean':
        return total_loss.mean()  # Mean over all elements
    elif reduction == 'sum':
        return total_loss.sum()  # Sum over all elements
    else:  # 'none'
        return total_loss.view(batch_size, seq_len)  # Reshape to original batch structure without reduction
