"""
Reference: https://github.com/oscarknagg/few-shot/blob/master/few_shot/matching.py
Data: 2023/02/13
"""

import torch
# import time # Unused import

EPSILON = 1e-8
# loss_fn = torch.nn.NLLLoss().cuda() # Will be initialized inside matching_loss

def matching_loss(prediction, target, n_support, n_way):
    '''
    Args:
    - prediction: the model output for a batch of samples. Shape: (batch_size, feature_dim)
    - target: ground truth for batch of samples. Shape: (batch_size,)
    - n-support: number of support samples per class
    - n-way: number of ways (classes)
    
    The Matching Networks loss calculation involves:
    1. Computing pairwise distances between query sample embeddings and support sample embeddings.
       (Typically cosine distance: $d(x_q, x_s) = 1 - \frac{x_q \cdot x_s}{\|x_q\| \|x_s\|}$)
    2. Calculating attention weights as softmax over negative distances:
       $a(x_q, x_s) = \frac{\exp(-d(x_q, x_s))}{\sum_{s' \in S} \exp(-d(x_q, x_{s'}))}$
    3. Predicting query sample labels as a weighted sum of one-hot encoded support labels:
       $\hat{y}_q = \sum_{s \in S} a(x_q, x_s) y_s^{\text{one-hot}}$
    4. Computing NLLLoss between predictions and true query labels:
       $L = \text{NLLLoss}(\log(\text{clamp}(\mathbf{Y}_{\text{pred}}, \epsilon, 1-\epsilon)), \mathbf{y}_{\text{true\_query}})$
    '''
    loss_fn = torch.nn.NLLLoss().to(prediction.device)

    prediction_cpu = prediction.to('cpu') # [class*(num_support + num_querry)]
    target_cpu = target.to('cpu') # [class*(num_support + num_querry)]

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support # number of samples per class - n_support

    def find_support_idxs(c):
        """
        Input a class 'c', return the indexes of support samples
        Fetch the first n_support samples as the support set per classes
        Return dtype: list
        """
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)
    
    def find_query_indxs(c):
        """
        Input a class 'c', return the indexes of query samples
        Return dtype: list
        """
        return target_cpu.eq(c).nonzero()[n_support:]

    # Get support and query set indexes
    support_indexes = torch.stack(list(map(find_support_idxs, classes))).view(-1)
    query_indexes = torch.stack(list(map(find_query_indxs, classes))).view(-1)

    # Fetch the support and query sample predictions
    support_samples = prediction[support_indexes] # 5-way 5-shot [50,64]
    query_samples = prediction[query_indexes] # 5-way 5-shot [50,64]

    # compute distances
    dists = pairwise_distances(query_samples, support_samples, 'cosine') # [50, 50]

    # Calculate "attention" as softmax over support-query distances
    attention = (-dists).softmax(dim=1) # [50, 50]

    # Calculate predictions as in equation (1) from Matching Networks
    # y_hat = \sum_{i=1}^{k} a(x_hat, x_i) y_i
    y_pred = matching_net_predictions(attention, n_support, n_way, n_query) # [50,10]

    # Calculated loss with negative log likelihood
    # Clip predictions for numerical stability
    clipped_y_pred = y_pred.clamp(EPSILON, 1 - EPSILON)
    # Loss calculation: $L = -\sum \log(P(y_j | x_j))$
    loss = loss_fn(clipped_y_pred.log(), target[query_indexes].to(device=prediction.device, dtype=torch.int64))

    y_hat = torch.max(y_pred, 1)[1].cpu().numpy() # max prediction indexs
    y_true = target[query_indexes].to(dtype=torch.int64).cpu().numpy()
    acc_query = (y_hat == y_true).sum() / len(y_true)


    return loss, acc_query


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Support samples (or class prototypes). A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples.
            'l2': Euclidean distance squared. $d(x_i, y_j) = \| x_i - y_j \|_2^2 = \sum_d (x_{id} - y_{jd})^2$
            'cosine': Cosine distance. $d(x_i, y_j) = 1 - \cos(x_i, y_j) = 1 - \frac{x_i \cdot y_j}{\|x_i\| \|y_j\|}$
            'dot': Dot product similarity (negative distance). $d(x_i, y_j) = - (x_i \cdot y_j)$
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))

def matching_net_predictions(attention: torch.Tensor, n: int, k: int, q: int) -> torch.Tensor:
    """Calculates Matching Network predictions based on equation (1) of the paper.
    The predictions are the weighted sum of the labels of the support set where the
    weights are the "attentions" (i.e. softmax over query-support distances) pointing
    from the query set samples to the support set samples.

    Equation: $\mathbf{Y}_{\text{pred}} = \mathbf{A} \cdot \mathbf{Y}_{\text{support\_onehot}}$
    where $\mathbf{A}$ is the attention matrix (shape $N_q \times N_s$) and
    $\mathbf{Y}_{\text{support\_onehot}}$ are one-hot labels of support samples (shape $N_s \times k$).

    # Arguments
        attention: torch.Tensor containing softmax over query-support distances.
            Should be of shape (q * k, k * n)
        n: Number of support set samples per class, n-shot (denoted $n_s$ in some contexts)
        k: Number of classes in the episode, k-way
        q: Number of query samples per-class

    # Returns
        y_pred: Predicted class probabilities for query samples. Shape (q * k, k)
    """
    if attention.shape != (q * k, k * n): # Example: 5-way 5-shot query, 5-way 5-shot support -> (q*k, k*n)
        raise(ValueError(f'Expecting attention Tensor to have shape (q * k, k * n) = ({q * k}, {k * n}), got {attention.shape}'))

    # Create one hot label vector for the support set
    # y_onehot = torch.zeros(k * n, k) # Original: 10-way 5-shot: [50, 10]

    # Unsqueeze to force y to be of shape (K*n, 1) as this
    # is needed for .scatter()
    # y = create_nshot_task_label(k, n).unsqueeze(-1) # y.shape = [k*n, 1]
    # y_onehot = y_onehot.scatter(1, y, 1)  # 生成onehot标签

    # Simplified one-hot label creation for support set
    support_labels = create_nshot_task_label(k, n) # Shape: (k * n)
    # $\mathbf{Y}_{\text{support\_onehot}}$
    y_onehot = torch.nn.functional.one_hot(support_labels, num_classes=k).to(attention.device, dtype=torch.float64)
    
    # $\mathbf{Y}_{\text{pred}} = \mathbf{A} \cdot \mathbf{Y}_{\text{support\_onehot}}$
    y_pred = torch.mm(attention.to(dtype=torch.float64), y_onehot) # Ensure mm is done in float64

    return y_pred

def create_nshot_task_label(k: int, num_samples_per_class: int) -> torch.Tensor:
    """Creates an n-shot task label.

    Label has the structure:
        $[0, \dots, 0, 1, \dots, 1, \dots, k-1, \dots, k-1]$
    where each class $i \in [0, k-1]$ is repeated `num_samples_per_class` times.
    Total length $k \times \text{num\_samples\_per\_class}$.

    # Arguments
        k: Number of classes in the n-shot classification task
        num_samples_per_class: Number of samples for each class (e.g., n_support or n_query)

    # Returns
        y: Label vector for n-shot task of shape [num_samples_per_class * k, ]
    """
    y = torch.arange(0, k, 1 / num_samples_per_class).long()
    return y

if __name__ == '__main__':
    def test_matching_loss():
        # Define test parameters
        n_support = 5  # Number of support samples per class
        n_way = 3      # Number of classes (ways)
        n_query = 2    # Number of query samples per class
        feature_dim = 64 # Dimension of the feature embeddings

        # Total samples per class
        samples_per_class = n_support + n_query
        # Total samples in the batch
        total_samples = n_way * samples_per_class

        # Create mock prediction tensor (features)
        # Shape: (total_samples, feature_dim)
        # Ensure predictions are on CUDA as loss_fn and other parts expect it
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("CUDA is available. Running test on CUDA.")
        else:
            print("CUDA not available. Skipping test as it requires CUDA.")
            return

        predictions = torch.randn(total_samples, feature_dim).to(device)

        # Create mock target tensor (labels)
        # Shape: (total_samples,)
        # Labels should be structured: [0...0, 1...1, ..., (n_way-1)...(n_way-1)]
        # Each class has 'samples_per_class' samples
        # target_labels = []
        # for i in range(n_way):
        #     target_labels.extend([i] * samples_per_class)
        # targets = torch.tensor(target_labels, dtype=torch.long).to(device)
        targets = torch.arange(n_way, device=device).repeat_interleave(samples_per_class)


        print(f"Test Parameters:")
        print(f"  n_support: {n_support}")
        print(f"  n_way: {n_way}")
        print(f"  n_query: {n_query}")
        print(f"  feature_dim: {feature_dim}")
        print(f"  Prediction shape: {predictions.shape}")
        print(f"  Target shape: {targets.shape}")
        # print(f"  Sample targets: {targets[:samples_per_class*2]}") # Print first few targets

        # Call the matching_loss function
        try:
            loss, acc_query = matching_loss(predictions, targets, n_support, n_way)
            print(f"\nCalculated Loss: {loss.item()}")
            print(f"Calculated Query Accuracy: {acc_query}")
        except Exception as e:
            print(f"An error occurred during matching_loss execution: {e}")
            import traceback
            traceback.print_exc()

    test_matching_loss()