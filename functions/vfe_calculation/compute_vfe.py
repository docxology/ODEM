import torch
from functions.vfe_calculation import compute_e_x,compute_e_y, compute_e_theta, compute_e_lambda, compute_log_det


def _quadratic_form(vector, matrix):
    flat = vector.reshape(-1)
    return torch.dot(flat, matrix @ flat)

def compute(gen_mu, gen_y, f, g, q_theta_mu, p_theta_eta, q_lambda_x_mu, p_lambda_x_eta,
            q_lambda_y_mu, p_lambda_y_eta, gen_pi_y, gen_pi_x, p_theta_pi, gen_mu_cov,
            q_theta_cov, p_lambda_x_pi, q_lambda_x_cov, p_lambda_y_pi, q_lambda_y_cov,
            device):
    """
    Computes the variational free energy (VFE) using:
    - Generative predictions and errors
    - Prior terms
    - Entropy (via logdet terms of posterior covariances)
    All computations are done in a numerically safe way using spm_logdet().
    """

    kx, dx, ky, dy = gen_mu.shape[0], gen_mu.shape[1], gen_y.shape[0], gen_y.shape[1]

    gen_e_y, gen_y_hat = compute_e_y.compute(gen_mu, gen_y, g)
    gen_e_x = compute_e_x.compute(gen_mu, f, q_theta_mu, device)

    e_theta = compute_e_theta.compute(q_theta_mu, p_theta_eta)
    e_lambda_x = compute_e_lambda.compute(q_lambda_x_mu, p_lambda_x_eta)
    e_lambda_y = compute_e_lambda.compute(q_lambda_y_mu, p_lambda_y_eta)

    # compute the quadratic terms in vfe
    quad_x = _quadratic_form(gen_e_x, gen_pi_x)
    quad_y = _quadratic_form(gen_e_y, gen_pi_y)

    quad_theta = _quadratic_form(e_theta, p_theta_pi)
    quad_lambda_x = _quadratic_form(e_lambda_x, p_lambda_x_pi)
    quad_lambda_y = _quadratic_form(e_lambda_y, p_lambda_y_pi)

    # Log-determinants (entropy and prior scale terms)
    logdet_pi_x         = compute_log_det.compute(gen_pi_x)
    logdet_pi_y         = compute_log_det.compute(gen_pi_y)

    logdet_pi_theta     = compute_log_det.compute(p_theta_pi)
    logdet_pi_lambda_x  = compute_log_det.compute(p_lambda_x_pi)
    logdet_pi_lambda_y  = compute_log_det.compute(p_lambda_y_pi)

    logdet_cov_theta    = compute_log_det.compute(q_theta_cov)
    logdet_cov_lambda_x = compute_log_det.compute(q_lambda_x_cov)
    logdet_cov_lambda_y = compute_log_det.compute(q_lambda_y_cov)
    logdet_cov_x        = compute_log_det.compute(gen_mu_cov)

    logdets = [
        logdet_pi_x, logdet_pi_y, logdet_pi_theta,
        logdet_pi_lambda_x, logdet_pi_lambda_y,
        logdet_cov_theta, logdet_cov_lambda_x,
        logdet_cov_lambda_y, logdet_cov_x
    ]

    # -----------------------
    # Accuracy (data term)
    # -----------------------
    accuracy = 0.5 * (-quad_y + logdet_pi_y - ky * dy *  torch.log(2 * torch.tensor(torch.pi)))
    # -----------------------
    # Complexity components
    # -----------------------
    complexity_x        = 0.5 * (quad_x        - logdet_pi_x        - logdet_cov_x)
    complexity_theta    = 0.5 * (quad_theta    - logdet_pi_theta    - logdet_cov_theta)
    complexity_lambda_x = 0.5 * (quad_lambda_x - logdet_pi_lambda_x - logdet_cov_lambda_x)
    complexity_lambda_y = 0.5 * (quad_lambda_y - logdet_pi_lambda_y - logdet_cov_lambda_y)

    complexity = complexity_x + complexity_theta + complexity_lambda_x + complexity_lambda_y

    # Total VFE
    vfe = complexity - accuracy

    return vfe, accuracy, complexity
