import sys
from dataclasses import dataclass, field

from tqdm import tqdm
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=9)

from algorithms import D_step, M_step, E_step
from functions import convert_tensor_lists_to_numpy
from functions.optimisation import compute_robins_monroe
from functions.vfe_calculation import (compute_D, compute_vfe,compute_generalised_precision,
                                       compute_e_y, compute_covariance_posteriors,update_gen_y, vfe_validity)
from functions.prior_initialisation import prior_theta, prior_lambda

from functions.noise_generation import add_noise
from functions import pick_cov
from odem.model_registry import resolve_process_build
from odem.numerics import ensure_finite_tensor, safe_precision_inverse


@dataclass
class DemTrace:
    vfe: list = field(default_factory=list)
    accuracy: list = field(default_factory=list)
    complexity: list = field(default_factory=list)
    theta: list = field(default_factory=list)
    lambda_x: list = field(default_factory=list)
    lambda_y: list = field(default_factory=list)
    cov_lambda_x: list = field(default_factory=list)
    cov_lambda_y: list = field(default_factory=list)
    cov_theta: list = field(default_factory=list)
    gen_sensations: list = field(default_factory=list)
    gen_predictions: list = field(default_factory=list)
    gen_x_estimates: list = field(default_factory=list)

    def append(
        self,
        *,
        vfe,
        accuracy,
        complexity,
        theta,
        lambda_x,
        lambda_y,
        cov_lambda_x,
        cov_lambda_y,
        cov_theta,
        gen_sensation,
        gen_prediction,
        gen_x_estimate,
    ) -> None:
        self.vfe.append(_detached(vfe))
        self.accuracy.append(_detached(accuracy))
        self.complexity.append(_detached(complexity))
        self.theta.append(_detached(theta))
        self.lambda_x.append(_detached(lambda_x))
        self.lambda_y.append(_detached(lambda_y))
        self.cov_lambda_x.append(_detached(cov_lambda_x))
        self.cov_lambda_y.append(_detached(cov_lambda_y))
        self.cov_theta.append(_detached(cov_theta))
        self.gen_sensations.append(_detached(gen_sensation))
        self.gen_predictions.append(_detached(gen_prediction))
        self.gen_x_estimates.append(_detached(gen_x_estimate))


def _detached(value):
    return value.detach().clone() if hasattr(value, "detach") else value


def start(kx, ky, f, g, f_name, gp_name,
            dt, T,
            E_theta, sigma_theta,
            E_pi_x, sigma_lambda_x,
            E_pi_y, sigma_lambda_y,
             nu_x, kappa_x,
             lambda_eta_adapt, lambda_eta_rate, lambda_eta_t_0, lambda_eta_gamma, lambda_beta,
             theta_eta_adapt, theta_eta_rate, theta_eta_t_0, theta_eta_gamma, EM_interval, theta_beta,
            initial_jitter, noise, carry_cov, device,
          tqdm_disable=True):

    # Load the selected generative process, determined by gp_name. In this script, we only have Lotka-Volterra as the GP.
    # GP is now a .py function that contains the Ordinaty Differential Equations (ODEs) describing the GP.

    gp_build = resolve_process_build(gp_name)
    # Solve the ODEs in GP by integrating them over a time span T and with step size dt.
    # x now holds the true external states of the world (i.e., true trajectories of the world)

    x_noise, y_noise = noise['x'], noise['y']

    # We won't need to pass gp_mode anymore since either lotka or lorenz is selected.
    x, x_clean, x_white_noise, x_sigma_schedule, x_colored_noise, x_context_lengths = gp_build(dt, T, x_noise)

    if gp_name == 'lotka' or gp_name == 'glv':
        y_noise_min_cap = 0
    else:
        y_noise_min_cap = None # This is for when the GP is NOT lotka

    # Add IID observation noise to noisy x, to build the noisy observations
    y, y_white_noise, y_sigma_schedule, y_colored_noise, y_context_lengths = add_noise.add(x, y_noise['y_wn_mu'], y_noise['y_wn_sigma'], y_noise['y_cn_kernel_size'], y_noise['y_cn_kernel_sigma'], min_cap= y_noise_min_cap)
    # The dimensions of x and y, which are both equal to 2 for a Lotka-Volterra process.
    dx, dy = x.shape[1], y.shape[1]

    # Parameter shapes
    param_shapes = {'glv': (3,), 'lorenz':(1,)}
    theta_shape = param_shapes[f_name]
    ######################################### Initialise priors #####################################################
    p_theta_eta, p_theta_pi = prior_theta.set(E_theta, sigma_theta, theta_shape)
    p_lambda_x_eta, p_lambda_x_pi, p_lambda_y_eta, p_lambda_y_pi = prior_lambda.set(E_pi_x, sigma_lambda_x,
                                                                                    E_pi_y, sigma_lambda_y)
    ############################ Initialise posterior expectations using the priors ################################
    q_theta_mu = p_theta_eta.detach().clone().requires_grad_(True)
    q_lambda_y_mu = p_lambda_y_eta.detach().clone().requires_grad_(True)
    q_lambda_x_mu = p_lambda_x_eta.detach().clone().requires_grad_(True)
    ################################# Initialise generalised precision matrices ####################################
    gen_pi_y = compute_generalised_precision.compute(q_lambda_y_mu, ky, dy, y_noise['y_h_value'], y_noise['y_lambda_value'])
    gen_pi_x = compute_generalised_precision.compute(q_lambda_x_mu, kx, dx, x_noise['x_h_value'], x_noise['x_lambda_value'])
    ############################ Initialise the generalised coordinates of motion ###################################
    # ky orders of generalised precisions y, y',y'', y''', ... &
    # kx orders of generalised dynamics x, x', x'', ...
    q_x_mu, gen_y = (torch.stack([torch.rand(dx) for _ in range(kx)]).requires_grad_(True),
                     torch.stack([torch.zeros(len(y[0])) for _ in range(ky)]),)
    ############################ Initialise the derivative matrix operator D ########################################
    D = compute_D.compute(kx, dx)
    ############################ Evaluate the posterior covariances using their modes ###############################
    gen_mu_cov, q_theta_cov, q_lambda_x_cov, q_lambda_y_cov = compute_covariance_posteriors.compute(
        q_x_mu, gen_y, g, f,
        q_theta_mu,
        p_theta_eta,
        p_theta_pi,
        p_lambda_x_eta,
        q_lambda_x_mu,
        p_lambda_x_pi,
        p_lambda_y_eta,
        q_lambda_y_mu,
        p_lambda_y_pi,
        y_noise['y_h_value'], y_noise['y_lambda_value'],
        x_noise['x_h_value'], x_noise['x_lambda_value'],
        'all', initial_jitter, device)
    # Guard covariances
    gen_mu_cov = pick_cov.pick(gen_mu_cov, None, dim=q_x_mu.numel(), device=q_x_mu.device, dtype=q_x_mu.dtype)
    q_theta_cov = pick_cov.pick(q_theta_cov, None, dim=q_theta_mu.numel(), device=q_theta_mu.device, dtype=q_theta_mu.dtype,
                           cov_prior=None)
    q_lambda_x_cov = pick_cov.pick(q_lambda_x_cov, None, dim=q_lambda_x_mu.numel(), device=q_lambda_x_mu.device,
                              dtype=q_lambda_x_mu.dtype)
    q_lambda_y_cov = pick_cov.pick(q_lambda_y_cov, None, dim=q_lambda_y_mu.numel(), device=q_lambda_y_mu.device,
                              dtype=q_lambda_y_mu.dtype)

    ####################################### Initialise trace storage ###############################################
    free_action = torch.zeros((), dtype=q_x_mu.dtype, device=q_x_mu.device)
    trace = DemTrace()
    # Posterior covariance traces for lambda_y and lambda_x.
    COV_LAMBDA_Y, COV_LAMBDA_X = [q_lambda_y_cov], [q_lambda_x_cov]
    COV_THETA = [q_theta_cov]
    # Store these for the visualisation of their evolution
    THETA, LAMBDA_X, LAMBDA_Y= [q_theta_mu], [q_lambda_x_mu], [q_lambda_y_mu]
    # These will hold the accumulated gradients for lambda and theta
    acc_grad_theta = torch.zeros_like(q_theta_mu)
    acc_grad_lambda_y, acc_grad_lambda_x = torch.zeros_like(q_lambda_y_mu), torch.zeros_like(q_lambda_x_mu)

    iterator = tqdm(y, desc="    ↳ Triple Estimation", unit="step", file=sys.stdout, disable=tqdm_disable)
    # The inference loop starts, which goes over each sensation/observation in y.
    for i, obs in enumerate(iterator):
        # Update gen_y given current observation obs. This serves as the ground truth for calculating the error term e_y
        # Given any new observation, gen_y needs to be updated. If ky>1, then not only the observation in gen_y needs to be
        # updated but also the estimates of velocity y', acceleration y'', jerk y''', etc. In our script ky=1, though.
        gen_y = update_gen_y.update(obs, gen_y, dt)
        ################################################# D step #######################################################
        q_x_mu, delta_s, ozaki_jitter = D_step.step(q_x_mu, gen_y, f, g, q_theta_mu, p_theta_eta, q_lambda_x_mu, p_lambda_x_eta,
                                  q_lambda_y_mu, p_lambda_y_eta,
                                  gen_pi_y, gen_pi_x, p_theta_pi, gen_mu_cov, q_theta_cov,
                                  p_lambda_x_pi, q_lambda_x_cov,
                                  p_lambda_y_pi, q_lambda_y_cov, kappa_x, nu_x, initial_jitter, D, i, device)
        ################################################################################################################
        # Estimate posterior covariances
        temp_gen_mu_cov, _, _, _ = compute_covariance_posteriors.compute(q_x_mu, gen_y, g, f,
                                                                    q_theta_mu,
                                                                    p_theta_eta,
                                                                    p_theta_pi,
                                                                    p_lambda_x_eta,
                                                                    q_lambda_x_mu,
                                                                    p_lambda_x_pi,
                                                                    p_lambda_y_eta,
                                                                    q_lambda_y_mu,
                                                                    p_lambda_y_pi,
                                                                     y_noise['y_h_value'],
                                                                     y_noise['y_lambda_value'],
                                                                     x_noise['x_h_value'],
                                                                     x_noise['x_lambda_value'],
                                                                    'x', initial_jitter, device)
        # Guard the covariance
        gen_mu_cov = pick_cov.pick(temp_gen_mu_cov, gen_mu_cov, dim=q_x_mu.numel(), device=q_x_mu.device, dtype=q_x_mu.dtype)
        # Recalculate VFE
        vfe, _, _ = compute_vfe.compute(q_x_mu, gen_y, f, g, q_theta_mu, p_theta_eta, q_lambda_x_mu, p_lambda_x_eta,
                                      q_lambda_y_mu, p_lambda_y_eta,
                                      gen_pi_y, gen_pi_x, p_theta_pi, gen_mu_cov, q_theta_cov,
                                      p_lambda_x_pi, q_lambda_x_cov,
                                      p_lambda_y_pi, q_lambda_y_cov, device)
        vfe_validity.check(vfe, step_info=f"After D-step before gradient accumulation={i}")
        # accumulate gradients using exponential smoothing
        # accumulate gradients w.r.t lambda_x
        g_x = torch.autograd.grad(vfe, q_lambda_x_mu, retain_graph=True)[0].detach()
        g_y = torch.autograd.grad(vfe, q_lambda_y_mu, retain_graph=True)[0].detach()

        acc_grad_lambda_x = lambda_beta * acc_grad_lambda_x + (1 - lambda_beta) * g_x
        acc_grad_lambda_y = lambda_beta * acc_grad_lambda_y + (1 - lambda_beta) * g_y

        g_theta_t = torch.autograd.grad(vfe, q_theta_mu)[0].detach()
        acc_grad_theta = theta_beta * acc_grad_theta + (1 - theta_beta) * g_theta_t
        ######################################## Entering the E and M steps ##########################################
        if (i + 1) % EM_interval == 0:
            # Compute the em_index for Robbins-Monro updating
            em_index = (i + 1) // EM_interval
            # ######################################## M step ##########################################################
            lambda_lr = compute_robins_monroe.compute(lambda_eta_rate, em_index, lambda_eta_t_0,
                                                      lambda_eta_gamma) if lambda_eta_adapt else lambda_eta_rate

            q_lambda_x_mu, q_lambda_y_mu = M_step.step(q_lambda_x_mu, q_lambda_y_mu, acc_grad_lambda_x, acc_grad_lambda_y, lambda_lr)

            ######################################### Generalised precisions #########################################
            gen_pi_y = compute_generalised_precision.compute(q_lambda_y_mu, ky, dy, y_noise['y_h_value'], y_noise['y_lambda_value'])
            gen_pi_x = compute_generalised_precision.compute(q_lambda_x_mu, kx, dx, x_noise['x_h_value'], x_noise['x_lambda_value'])
            ######################################### Lambda posterior covariances ###################################
            _, _, temp_q_lambda_x_cov, temp_q_lambda_y_cov = compute_covariance_posteriors.compute(
                q_x_mu, gen_y, g, f,
                q_theta_mu,
                p_theta_eta,
                p_theta_pi,
                p_lambda_x_eta,
                q_lambda_x_mu,
                p_lambda_x_pi,
                p_lambda_y_eta,
                q_lambda_y_mu,
                p_lambda_y_pi,
                y_noise['y_h_value'],
                y_noise['y_lambda_value'],
                x_noise['x_h_value'],
                x_noise['x_lambda_value'],
                'lambda', initial_jitter, device)

            # Safe-guard covariances
            temp_q_lambda_x_cov = pick_cov.pick(temp_q_lambda_x_cov, q_lambda_x_cov, dim=q_lambda_x_mu.numel(),
                                           device=q_lambda_x_mu.device, dtype=q_lambda_x_mu.dtype)
            temp_q_lambda_y_cov = pick_cov.pick(temp_q_lambda_y_cov, q_lambda_y_cov, dim=q_lambda_y_mu.numel(),
                                           device=q_lambda_y_mu.device, dtype=q_lambda_y_mu.dtype)
            q_lambda_x_cov = temp_q_lambda_x_cov
            q_lambda_y_cov = temp_q_lambda_y_cov
            ######################################### Update prior expectations ######################################
            # (.detach() is needed to avoid carrying the computational graph
            p_lambda_x_eta = q_lambda_x_mu.detach().clone()
            p_lambda_y_eta = q_lambda_y_mu.detach().clone()
            ######################################### Update prior precisions ########################################
            if carry_cov:
                p_lambda_x_pi = safe_precision_inverse(q_lambda_x_cov, previous_precision=p_lambda_x_pi, phase="M-step lambda_x", timestep=i)
                p_lambda_y_pi = safe_precision_inverse(q_lambda_y_cov, previous_precision=p_lambda_y_pi, phase="M-step lambda_y", timestep=i)
            ################################## E step  ###############################################################
            theta_lr = compute_robins_monroe.compute(theta_eta_rate, em_index, theta_eta_t_0,
                                                     theta_eta_gamma) if theta_eta_adapt else theta_eta_rate
            q_theta_mu = E_step.step(q_theta_mu, acc_grad_theta, theta_lr)
            ####################################### Theta posterior covariances ###################################
            _, temp_q_theta_cov, _, _ = compute_covariance_posteriors.compute(
                q_x_mu, gen_y, g, f,
                q_theta_mu,
                p_theta_eta,
                p_theta_pi,
                p_lambda_x_eta,
                q_lambda_x_mu,
                p_lambda_x_pi,
                p_lambda_y_eta,
                q_lambda_y_mu,
                p_lambda_y_pi,
                y_noise['y_h_value'],
                y_noise['y_lambda_value'],
                x_noise['x_h_value'],
                x_noise['x_lambda_value'],
                'theta', initial_jitter, device)
            # Safe-guard covariances
            temp_q_theta_cov = pick_cov.pick(temp_q_theta_cov, q_theta_cov, dim=q_theta_mu.numel(), device=q_theta_mu.device,
                                        dtype=q_theta_mu.dtype)
            q_theta_cov = temp_q_theta_cov
            ######################################### Update prior expectations ######################################
            p_theta_eta = q_theta_mu.detach().clone()
            ######################################### Update prior precisions ########################################
            if carry_cov:
                p_theta_pi = safe_precision_inverse(q_theta_cov, previous_precision=p_theta_pi, phase="E-step theta", timestep=i)
            ######################################### Reset accumulators #############################################
            acc_grad_theta = torch.zeros_like(q_theta_mu, device=q_theta_mu.device, dtype=q_theta_mu.dtype)
            acc_grad_lambda_y = torch.zeros_like(q_lambda_y_mu, device=q_lambda_y_mu.device, dtype=q_lambda_y_mu.dtype)
            acc_grad_lambda_x = torch.zeros_like(q_lambda_x_mu, device=q_lambda_x_mu.device, dtype=q_lambda_x_mu.dtype)

        # Get gen_y_hat for plotting the predicted sensations
        _, gen_y_hat = compute_e_y.compute(q_x_mu, gen_y, g)
        # Final evaluation of vfe
        vfe, accuracy, complexity = compute_vfe.compute(q_x_mu, gen_y, f, g, q_theta_mu, p_theta_eta, q_lambda_x_mu, p_lambda_x_eta,
                                  q_lambda_y_mu, p_lambda_y_eta,
                                  gen_pi_y, gen_pi_x, p_theta_pi, gen_mu_cov, q_theta_cov,
                                  p_lambda_x_pi, q_lambda_x_cov,
                                  p_lambda_y_pi, q_lambda_y_cov, device)
        vfe_validity.check(vfe, step_info=f"Final vfe calculation t={i}")
        ensure_finite_tensor(vfe, name="vfe", phase="final VFE", timestep=i)
        ensure_finite_tensor(accuracy, name="accuracy", phase="final VFE", timestep=i)
        ensure_finite_tensor(complexity, name="complexity", phase="final VFE", timestep=i)

        # Accumulate vfe into free action
        free_action += vfe
        ensure_finite_tensor(free_action, name="free_action", phase="accumulation", timestep=i)

        if i % 10 == 0 and not tqdm_disable:
            print('\nFree Action=%.2f, VFE: %.2f' % (free_action.detach(), vfe.detach()))

        ensure_finite_tensor(q_lambda_x_cov, name="q_lambda_x_cov", phase="trace", timestep=i)
        ensure_finite_tensor(q_lambda_y_cov, name="q_lambda_y_cov", phase="trace", timestep=i)
        ensure_finite_tensor(q_theta_cov, name="q_theta_cov", phase="trace", timestep=i)
        trace.append(
            vfe=vfe,
            accuracy=accuracy,
            complexity=complexity,
            theta=q_theta_mu,
            lambda_x=q_lambda_x_mu,
            lambda_y=q_lambda_y_mu,
            cov_lambda_x=q_lambda_x_cov,
            cov_lambda_y=q_lambda_y_cov,
            cov_theta=q_theta_cov,
            gen_sensation=gen_y,
            gen_prediction=gen_y_hat,
            gen_x_estimate=q_x_mu,
        )

    tensor_lists = [
        trace.vfe, trace.accuracy, trace.complexity, trace.theta, trace.lambda_x, trace.lambda_y,
        trace.cov_lambda_x, trace.cov_lambda_y, trace.cov_theta,
        trace.gen_sensations, trace.gen_x_estimates, trace.gen_predictions,
        x, y
    ]

    # Convert each list of tensor to a list of numpy arrays
    VFE, ACCURACY, COMPLEXITY, THETA, LAMBDA_X, LAMBDA_Y, COV_LAMBDA_X, COV_LAMBDA_Y, COV_THETA,\
        gen_sensations, gen_x_estimates, gen_predictions, \
        x, y = convert_tensor_lists_to_numpy.convert(tensor_lists)

    # Evaluate the performance through MSE and Free Action
    # gen_x_estimates[:,0,:] holds all the estimates for x (i.e., mu_x) ignoring mu_x_dot, mu_x_dotdot, etc.
    x_estimates = gen_x_estimates[:, 0, :]


    mse = np.mean((x - x_estimates) ** 2) #--> changed x_noisy to x. Important! True states ARE the noisy states


    return (VFE, ACCURACY, COMPLEXITY, gen_sensations, gen_predictions, x_clean, x, y, gen_x_estimates, THETA
            ,LAMBDA_X, LAMBDA_Y, COV_LAMBDA_X, COV_LAMBDA_Y, COV_THETA,
            y_white_noise, y_sigma_schedule, y_colored_noise, y_context_lengths,
            x_white_noise, x_sigma_schedule, x_colored_noise, x_context_lengths,
            p_theta_eta, p_theta_pi, free_action, mse)
