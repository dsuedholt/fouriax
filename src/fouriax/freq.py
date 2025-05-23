from functools import partial

import jax
import jax.numpy as jnp
import scipy


def spectral_convergence_loss(x_mag, y_mag, eps=1e-8):
    """
    Calculate the spectral convergence loss.

    Args:
        x_mag (array): The magnitude spectrum of the first signal.
        y_mag (array): The magnitude spectrum of the second signal.

    Returns:
        The spectral convergence loss.
    """
    numerator = jnp.linalg.norm(y_mag - x_mag, ord="fro")
    denominator = jnp.linalg.norm(y_mag, ord="fro") + eps
    loss = numerator / denominator
    return loss


def stft_magnitude_loss(
    x_mag, y_mag, log=True, distance="L1", reduction="mean", log_fac=1.0, log_eps=1e-8
):
    """
    Calculate the STFT magnitude loss.

    Log-magnitudes are calculated with `log(log_fac*x + log_eps)`, where `log_fac` controls the
    compression strength (larger value results in more compression), and `log_eps` can be used
    to control the range of the compressed output values (e.g., `log_eps>=1` ensures positive
    output values). The default values `log_fac=1` and `log_eps=0` correspond to plain log-compression.

    Args:
        x_mag (array): The magnitude spectrum of the first signal.
        y_mag (array): The magnitude spectrum of the second signal.
        log (bool): Whether to log-scale the STFT magnitudes.
        distance (str): Distance function ["L1", "L2"].
        reduction (str): Reduction of the loss elements ["mean", "sum", "none"].
        log_eps (float, optional): Constant value added to the magnitudes before evaluating the logarithm.
            Default: 1e-8
        log_fac (float, optional): Constant multiplication factor for the magnitudes before evaluating the logarithm.
            Default: 1.0
    Returns:
        The STFT magnitude loss.
    """
    if log:
        x_mag = jnp.log(
            log_fac * x_mag + log_eps
        )  # Adding a small value to avoid log(0)
        y_mag = jnp.log(log_fac * y_mag + log_eps)

    if distance == "L1":
        # L1 Loss (Mean Absolute Error)
        loss = jnp.abs(x_mag - y_mag)
    elif distance == "L2":
        # L2 Loss (Mean Squared Error)
        loss = (x_mag - y_mag) ** 2
    else:
        raise ValueError(f"Invalid distance: '{distance}'.")

    if reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "sum":
        return jnp.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: '{reduction}'.")


def stft_loss(
    inputs,
    target,
    fft_size: int = 1024,
    hop_size: int = 256,
    win_length: int = 1024,
    window: str | jnp.ndarray = "hann",
    w_sc=1.0,
    w_log_mag=1.0,
    w_lin_mag=0.0,
    w_phs=0.0,
    scale=None,
    perceptual_weighting=None,
    scale_invariance=False,
    eps=1e-8,
    log_eps=1e-8,
    log_fac=1.0,
    output="loss",
    reduction="mean",
    mag_distance="L1",
    axis=1,
    undo_window_norm=True,
):
    """
    Calculate the STFT loss.
    """
    assert inputs.shape == target.shape
    assert -inputs.ndim <= axis < inputs.ndim

    inputs = jnp.reshape(inputs, (-1, inputs.shape[axis]))
    target = jnp.reshape(target, (-1, target.shape[axis]))

    if perceptual_weighting is not None:
        inputs, target = perceptual_weighting(inputs), perceptual_weighting(target)

    if isinstance(window, str):
        win = jnp.array(scipy.signal.get_window(window, win_length, fftbins=True))

    def stft(x):
        noverlap = win_length - hop_size
        _, _, out = jax.scipy.signal.stft(
            x,
            window=win,
            nperseg=win_length,
            noverlap=noverlap,
            nfft=fft_size,
            axis=axis,
            boundary="even",
            padded=False,
        )

        # unlike torch.stft, jax.scipy.signal.stft divides the result by the sum of the window
        # this can't be disabled through the function signature
        # so we undo it by default to be consistent with auraloss
        if undo_window_norm:
            out *= win.sum()

        return jnp.reshape(out, (out.shape[-2], -1))

    inputs_stft = stft(inputs)
    target_stft = stft(target)

    inputs_phs, target_phs = None, None
    if w_phs:
        inputs_phs = jnp.angle(inputs_stft)
        target_phs = jnp.angle(target_stft)

    def mag(x):
        return jnp.sqrt(jnp.clip((x.real**2) + (x.imag**2), min=eps))

    inputs_mag = mag(inputs_stft)
    target_mag = mag(target_stft)

    # Apply scaling (e.g., Mel, Chroma) if required
    if scale is not None:
        inputs_mag = jnp.matmul(scale, inputs_mag)
        target_mag = jnp.matmul(scale, target_mag)

    if scale_invariance:
        alpha = (inputs_mag * target_mag).sum(axis=(-2, -1)) / (target_mag**2).sum(
            axis=(-2, -1)
        )
        target_mag = target_mag * jnp.expand_dims(alpha, axis=-1)

    sc_mag_loss = (
        spectral_convergence_loss(inputs_mag, target_mag) * w_sc if w_sc else 0.0
    )
    log_mag_loss = (
        stft_magnitude_loss(
            inputs_mag,
            target_mag,
            log=True,
            reduction=reduction,
            distance=mag_distance,
            log_fac=log_fac,
            log_eps=log_eps,
        )
        * w_log_mag
        if w_log_mag
        else 0.0
    )
    lin_mag_loss = (
        stft_magnitude_loss(
            inputs_mag,
            target_mag,
            log=False,
            reduction=reduction,
            distance=mag_distance,
        )
        * w_lin_mag
        if w_lin_mag
        else 0.0
    )
    phs_loss = ((inputs_phs - target_phs) ** 2).mean() * w_phs if w_phs else 0.0

    # Combine loss components
    total_loss = sc_mag_loss + log_mag_loss + lin_mag_loss + phs_loss

    # Apply reduction (mean, sum)
    if reduction == "mean":
        total_loss = jnp.mean(total_loss)
    elif reduction == "sum":
        total_loss = jnp.sum(total_loss)

    # Return based on the output type
    if output == "loss":
        return total_loss
    elif output == "full":
        return total_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss


def multi_resolution_stft_loss(
    inputs,
    target,
    fft_sizes=(1024, 2048, 512),
    hop_sizes=(120, 240, 50),
    win_lengths=(600, 1200, 240),
    window="hann",
    w_sc=1.0,
    w_log_mag=1.0,
    w_lin_mag=0.0,
    w_phs=0.0,
    scale=None,
    perceptual_weighting=None,
    scale_invariance=False,
    eps=1e-8,
    log_eps=1e-8,
    log_fac=1.0,
    output="loss",
    reduction="mean",
    mag_distance="L1",
    axis=1,
    undo_window_norm=True,
):
    mrstft_loss = 0.0
    sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = [], [], [], []
    loss_fn = partial(
        stft_loss,
        inputs=inputs,
        target=target,
        window=window,
        w_sc=w_sc,
        w_log_mag=w_log_mag,
        w_lin_mag=w_lin_mag,
        w_phs=w_phs,
        scale=scale,
        perceptual_weighting=perceptual_weighting,
        scale_invariance=False,
        eps=eps,
        log_eps=log_eps,
        log_fac=log_fac,
        output=output,
        reduction=reduction,
        mag_distance=mag_distance,
        axis=axis,
        undo_window_norm=undo_window_norm,
    )

    for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
        if output == "full":
            tmp_loss = loss_fn(fft_size=fs, hop_size=hs, win_length=wl)
            mrstft_loss += tmp_loss[0]
            sc_mag_loss.append(tmp_loss[1])
            log_mag_loss.append(tmp_loss[2])
            lin_mag_loss.append(tmp_loss[3])
            phs_loss.append(tmp_loss[4])
        else:
            mrstft_loss += loss_fn(fft_size=fs, hop_size=hs, win_length=wl)

    mrstft_loss /= len(fft_sizes)

    if output == "loss":
        return mrstft_loss
    else:
        return mrstft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss


def sum_and_difference_stft_loss(
    inputs,
    target,
    fft_sizes=(1024, 2048, 512),
    hop_sizes=(120, 240, 50),
    win_lengths=(600, 1200, 240),
    w_sum=1.0,
    w_diff=1.0,
    output="loss",
    ch_axis=1,
    **kwargs,
):
    assert inputs.shape[ch_axis] == 2

    loss_fn = partial(
        multi_resolution_stft_loss,
        fft_sizes=fft_sizes,
        hop_sizes=hop_sizes,
        win_lengths=win_lengths,
        **kwargs,
    )

    def sd(x):
        x_sum = jnp.sum(x, axis=ch_axis)
        x_diff = jnp.diff(x, axis=ch_axis).squeeze(axis=ch_axis)
        return x_sum, x_diff

    inputs_sum, inputs_diff = sd(inputs)
    target_sum, target_diff = sd(target)

    sum_loss = loss_fn(inputs_sum, target_sum)
    diff_loss = loss_fn(inputs_diff, target_diff)

    loss = (sum_loss * w_sum + diff_loss * w_diff) / 2

    if output == "loss":
        return loss
    elif output == "full":
        return loss, sum_loss, diff_loss
