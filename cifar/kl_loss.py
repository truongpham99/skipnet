import torch

class DirichletKLLoss:
    """
    Can be applied to any model which returns logits

    """

    def __init__(self, target_concentration=1e3, concentration=1.0, reverse=True):
        """
        :param target_concentration: The concentration parameter for the
        target class (if provided)
        :param concentration: The 'base' concentration parameters for
        non-target classes.
        """
        self.target_concentration = torch.tensor(target_concentration,
                                                 dtype=torch.float32)
        self.concentration = concentration
        self.reverse = reverse

    def __call__(self, logits, labels, reduction='mean'):
        alphas = torch.exp(logits)
        return self.forward(alphas, labels, reduction=reduction)

    def forward(self, alphas, labels, reduction='mean'):
        loss = self.compute_loss(alphas, labels)

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'none':
            return loss
        else:
            raise NotImplementedError

    def compute_loss(self, alphas, labels = None):
        """
        :param alphas: The alpha parameter outputs from the model
        :param labels: Optional. The target labels indicating the correct
        class.

        The loss creates a set of target alpha (concentration) parameters
        with all values set to self.concentration, except for the correct
        class (if provided), which is set to self.target_concentration
        :return: an array of per example loss
        """
        # TODO: Need to make sure this actually works right...
        # todo: so that concentration is either fixed, or on a per-example setup
        # Create array of target (desired) concentration parameters
        target_alphas = torch.ones_like(alphas) * self.concentration
        if labels is not None:
            target_alphas += torch.zeros_like(alphas).scatter_(1, labels[:, None],
                                                               self.target_concentration)

        if self.reverse:
            loss = dirichlet_reverse_kl_divergence(alphas=alphas, target_alphas=target_alphas)
        else:
            loss = dirichlet_kl_divergence(alphas=alphas, target_alphas=target_alphas)
        return loss


def dirichlet_kl_divergence(alphas, target_alphas, precision=None, target_precision=None,
                            epsilon=1e-8):
    """
    This function computes the Forward KL divergence between a model Dirichlet distribution
    and a target Dirichlet distribution based on the concentration (alpha) parameters of each.

    :param alphas: Tensor containing concentation parameters of model. Expected shape is batchsize X num_classes.
    :param target_alphas: Tensor containing target concentation parameters. Expected shape is batchsize X num_classes.
    :param precision: Optional argument. Can pass in precision of model. Expected shape is batchsize X 1
    :param target_precision: Optional argument. Can pass in target precision. Expected shape is batchsize X 1
    :param epsilon: Smoothing factor for numercal stability. Default value is 1e-8
    :return: Tensor for Batchsize X 1 of forward KL divergences between target Dirichlet and model
    """
    if not precision:
        precision = torch.sum(alphas, dim=1, keepdim=True)
    if not target_precision:
        target_precision = torch.sum(target_alphas, dim=1, keepdim=True)

    precision_term = torch.lgamma(target_precision) - torch.lgamma(precision)
    assert torch.all(torch.isfinite(precision_term)).item()
    alphas_term = torch.sum(torch.lgamma(alphas + epsilon) - torch.lgamma(target_alphas + epsilon)
                            + (target_alphas - alphas) * (torch.digamma(target_alphas + epsilon)
                                                          - torch.digamma(
                target_precision + epsilon)), dim=1, keepdim=True)
    assert torch.all(torch.isfinite(alphas_term)).item()

    cost = torch.squeeze(precision_term + alphas_term)
    return cost


def dirichlet_reverse_kl_divergence(alphas, target_alphas, precision=None, target_precision=None,
                                    epsilon=1e-8):
    """
    This function computes the Reverse KL divergence between a model Dirichlet distribution
    and a target Dirichlet distribution based on the concentration (alpha) parameters of each.

    :param alphas: Tensor containing concentation parameters of model. Expected shape is batchsize X num_classes.
    :param target_alphas: Tensor containing target concentation parameters. Expected shape is batchsize X num_classes.
    :param precision: Optional argument. Can pass in precision of model. Expected shape is batchsize X 1
    :param target_precision: Optional argument. Can pass in target precision. Expected shape is batchsize X 1
    :param epsilon: Smoothing factor for numercal stability. Default value is 1e-8
    :return: Tensor for Batchsize X 1 of reverse KL divergences between target Dirichlet and model
    """
    return dirichlet_kl_divergence(alphas=target_alphas, target_alphas=alphas,
                                   precision=target_precision,
                                   target_precision=precision, epsilon=epsilon)

def dirichlet_prior_network_uncertainty(logits, epsilon=1e-10):
    """

    :param logits:
    :param epsilon:
    :return:
    """

    logits = np.asarray(logits, dtype=np.float64)
    alphas = np.exp(logits)
    alpha0 = np.sum(alphas, axis=1, keepdims=True)
    probs = alphas / alpha0

    conf = np.max(probs, axis=1)

    entropy_of_exp = -np.sum(probs * np.log(probs + epsilon), axis=1)
    expected_entropy = -np.sum((alphas / alpha0) * (digamma(alphas + 1) - digamma(alpha0 + 1.0)),
                               axis=1)
    mutual_info = entropy_of_exp - expected_entropy

    epkl = np.squeeze((alphas.shape[1] - 1.0) / alpha0)

    dentropy = np.sum(gammaln(alphas) - (alphas - 1.0) * (digamma(alphas) - digamma(alpha0)),
                      axis=1, keepdims=True) \
               - gammaln(alpha0)

    uncertainty = {'confidence': conf,
                   'entropy_of_expected': entropy_of_exp,
                   'expected_entropy': expected_entropy,
                   'mutual_information': mutual_info,
                   'EPKL': epkl,
                   'differential_entropy': np.squeeze(dentropy),
                   }

    return uncertainty
