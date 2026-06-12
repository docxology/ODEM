import torch
import torch.autograd.functional as F
from functions.vfe_calculation import (compute_e_y, compute_e_x, compute_e_theta,
                                       compute_e_lambda,compute_generalised_precision, compute_safe_inverse_from_hessian)


def _quadratic_form(vector, matrix):
    flat = vector.reshape(-1)
    return torch.dot(flat, matrix @ flat)


def compute_joint(gen_mu, gen_y, g, f, q_theta_mu, p_theta_eta,
                  y_h_value, y_lambda_value,
                  x_h_value, x_lambda_value,
                  p_theta_pi,
                  p_lambda_x_eta, q_lambda_x_mu, p_lambda_x_pi,
                  p_lambda_y_eta, q_lambda_y_mu, p_lambda_y_pi, device):

    kx, dx, ky, dy = gen_mu.shape[0], gen_mu.shape[1], gen_y.shape[0], gen_y.shape[1]

    # Construct generalised precision matrices over the data p_y and over the states p_x accordingly
    gen_pi_y = compute_generalised_precision.compute(q_lambda_y_mu, ky, dy, y_h_value, y_lambda_value)
    gen_pi_x = compute_generalised_precision.compute(q_lambda_x_mu, kx, dx, x_h_value, x_lambda_value)

    # Compute the different latent variables
    gen_e_y, _ = compute_e_y.compute(gen_mu, gen_y, g)
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


    return 0.5 * (quad_x + quad_y + quad_theta + quad_lambda_x + quad_lambda_y)


def compute(q_x_mu, gen_y, g, f, q_theta_mu, p_theta_eta, p_theta_pi, p_lambda_x_eta,
            q_lambda_x_mu,
            p_lambda_x_pi, p_lambda_y_eta,
            q_lambda_y_mu,
            p_lambda_y_pi,
            y_h_value, y_lambda_value,
            x_h_value, x_lambda_value,
            key, initial_jitter, device):
    gen_mu_cov, q_theta_cov, q_lambda_x_cov, q_lambda_y_cov = None,None,None,None
    if key in ['x', 'all']:
        # Flatten gen_mu
        gen_mu_flat = q_x_mu.view(-1)

        # Compute Hessians w.r.t. each element in the flattened gen_mu
        hessian_q = F.hessian(
            lambda q: compute_joint(
                q.view(q_x_mu.shape),  # Reshape g back to the original shape of gen_mu
                gen_y, g, f,
                q_theta_mu, p_theta_eta, y_h_value, y_lambda_value,
                  x_h_value, x_lambda_value, p_theta_pi,
                p_lambda_x_eta, q_lambda_x_mu, p_lambda_x_pi,
                p_lambda_y_eta, q_lambda_y_mu, p_lambda_y_pi, device),
            gen_mu_flat
        )

        gen_mu_cov = compute_safe_inverse_from_hessian.compute(hessian_q)


    if key in ['theta', 'all']:
        hessian_q = F.hessian(
            lambda q: compute_joint(
                q_x_mu, gen_y, g, f,
                q,  # No reshaping needed as q is already a vector tensor matching q_theta_mu
                p_theta_eta, y_h_value, y_lambda_value,
                  x_h_value, x_lambda_value, p_theta_pi,
                p_lambda_x_eta, q_lambda_x_mu, p_lambda_x_pi,
                p_lambda_y_eta, q_lambda_y_mu, p_lambda_y_pi, device
            ),
            q_theta_mu
        )

        q_theta_cov = compute_safe_inverse_from_hessian.compute(hessian_q)

    if key in ['lambda', 'all']:
        hessian_q = F.hessian(
            lambda q: compute_joint(
                q_x_mu, gen_y, g, f,
                q_theta_mu, p_theta_eta, y_h_value, y_lambda_value,
                  x_h_value, x_lambda_value, p_theta_pi,
                p_lambda_x_eta, q, p_lambda_x_pi,
                p_lambda_y_eta, q_lambda_y_mu, p_lambda_y_pi, device
            ),
            q_lambda_x_mu
        )

        q_lambda_x_cov = compute_safe_inverse_from_hessian.compute(hessian_q)

    if key in ['lambda', 'all']:
        hessian_q = F.hessian(
            lambda q: compute_joint(
                q_x_mu, gen_y, g, f,
                q_theta_mu, p_theta_eta, y_h_value, y_lambda_value,
                  x_h_value, x_lambda_value, p_theta_pi,
                p_lambda_x_eta, q_lambda_x_mu, p_lambda_x_pi,
                p_lambda_y_eta, q, p_lambda_y_pi, device
            ),
            q_lambda_y_mu
        )

        q_lambda_y_cov = compute_safe_inverse_from_hessian.compute(hessian_q)

    return gen_mu_cov, q_theta_cov, q_lambda_x_cov, q_lambda_y_cov
