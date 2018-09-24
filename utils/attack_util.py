import numpy as np

# define indexes for face regions

left_eye_idx = range(37 - 1, 42)
right_eye_idx = range(43 - 1, 48)
mouth_idx = range(49 - 1, 68)
nose_idx = range(28 - 1, 36)
jaw_idx = range(1 - 1, 17)
left_eyebrow_idx = range(18 - 1, 22)
right_eyebrow_idx = range(23 - 1, 27)




def purturb_GFLM(lnd_adversarial, grad, epsilon):

    # perturb mouth position and scale
    part_idx = mouth_idx
    part_lnd = lnd_adversarial[part_idx, :]
    part_lnd_mean = np.mean(part_lnd, 0, keepdims=True)
    part_f = np.sign(grad[part_idx, :]) * epsilon
    part_f_mean = np.mean(part_f, axis=0, keepdims=True)
    beta = part_f_mean + part_lnd_mean
    part_lnd_zero_mean = part_lnd - part_lnd_mean
    alpha = np.mean(part_lnd_zero_mean * (part_lnd + part_f), axis=0, keepdims=True) / np.mean(part_lnd_zero_mean ** 2,
                                                                                               axis=0, keepdims=True)
    alpha = np.clip(alpha, a_min=0.95, a_max=1. / 0.95)
    lnd_adversarial[part_idx, :] = part_lnd_zero_mean * alpha + beta

    # perturb nose position and scale
    part_idx = nose_idx
    part_lnd = lnd_adversarial[part_idx, :]
    part_lnd_mean = np.mean(part_lnd, 0, keepdims=True)
    part_f = np.sign(grad[part_idx, :]) * epsilon
    part_f_mean = np.mean(part_f, axis=0, keepdims=True)
    beta = part_f_mean + part_lnd_mean
    part_lnd_zero_mean = part_lnd - part_lnd_mean
    alpha = np.mean(part_lnd_zero_mean * (part_lnd + part_f), axis=0, keepdims=True) / np.mean(part_lnd_zero_mean ** 2,
                                                                                               axis=0, keepdims=True)
    alpha = np.clip(alpha, a_min=0.95, a_max=1. / 0.95)
    lnd_adversarial[part_idx, :] = part_lnd_zero_mean * alpha + beta

    # perturb eyes

    part_idx = right_eye_idx + left_eye_idx
    part_lnd = lnd_adversarial[part_idx, :]
    part_lnd_mean = np.mean(part_lnd, 0, keepdims=True)
    part_f = np.sign(grad[part_idx, :]) * epsilon
    part_f_mean = np.mean(part_f, axis=0, keepdims=True)
    beta = part_f_mean + part_lnd_mean
    part_lnd_zero_mean = part_lnd - part_lnd_mean
    alpha = np.mean(part_lnd_zero_mean * (part_lnd + part_f), axis=0, keepdims=True) / np.mean(part_lnd_zero_mean ** 2,
                                                                                               axis=0, keepdims=True)
    alpha = np.clip(alpha, a_min=0.95, a_max=1. / 0.95)
    lnd_adversarial[part_idx, :] = part_lnd_zero_mean * alpha + beta

    # perturb eyebrows

    part_idx = right_eyebrow_idx + left_eyebrow_idx
    part_lnd = lnd_adversarial[part_idx, :]
    part_lnd_mean = np.mean(part_lnd, 0, keepdims=True)
    part_f = np.sign(grad[part_idx, :]) * epsilon
    part_f_mean = np.mean(part_f, axis=0, keepdims=True)
    beta = part_f_mean + part_lnd_mean
    part_lnd_zero_mean = part_lnd - part_lnd_mean
    alpha = np.mean(part_lnd_zero_mean * (part_lnd + part_f), axis=0, keepdims=True) / np.mean(part_lnd_zero_mean ** 2,
                                                                                               axis=0, keepdims=True)
    alpha = np.clip(alpha, a_min=0.95, a_max=1. / 0.95)
    lnd_adversarial[part_idx, :] = part_lnd_zero_mean * alpha + beta

    #
    # # perturb left eye position and scale
    # part_idx = left_eye_idx
    # part_lnd = p_lnd[part_idx, :]
    # part_lnd_mean = np.mean(part_lnd, 0, keepdims=True)
    # part_f = np.sign(grad_[part_idx, :]) * epsilon
    # part_f_mean = np.mean(part_f, axis=0, keepdims=True)
    # beta = part_f_mean + part_lnd_mean
    # part_lnd_zero_mean = part_lnd-part_lnd_mean
    # alpha = np.mean(part_lnd_zero_mean*(part_lnd+part_f), axis=0, keepdims=True)/np.mean(part_lnd_zero_mean**2, axis=0, keepdims=True)
    # print alpha
    # p_lnd[part_idx, :] = part_lnd_zero_mean*alpha + beta

    # # perturb right eyebrow position and scale
    # part_idx = right_eyebrow_idx
    # part_lnd = p_lnd[part_idx, :]
    # part_lnd_mean = np.mean(part_lnd, 0, keepdims=True)
    # part_f = np.sign(grad_[part_idx, :]) * epsilon
    # part_f_mean = np.mean(part_f, axis=0, keepdims=True)
    # beta = part_f_mean + part_lnd_mean
    # part_lnd_zero_mean = part_lnd-part_lnd_mean
    # alpha = np.mean(part_lnd_zero_mean*(part_lnd+part_f), axis=0, keepdims=True)/np.mean(part_lnd_zero_mean**2, axis=0, keepdims=True)
    # print alpha
    # p_lnd[part_idx, :] = part_lnd_zero_mean*alpha + beta
    #
    # # perturb left eyebrow position and scale
    # part_idx = left_eyebrow_idx
    # part_lnd = p_lnd[part_idx, :]
    # part_lnd_mean = np.mean(part_lnd, 0, keepdims=True)
    # part_f = np.sign(grad_[part_idx, :]) * epsilon
    # part_f_mean = np.mean(part_f, axis=0, keepdims=True)
    # beta = part_f_mean + part_lnd_mean
    # part_lnd_zero_mean = part_lnd-part_lnd_mean
    # alpha = np.mean(part_lnd_zero_mean*(part_lnd+part_f), axis=0, keepdims=True)/np.mean(part_lnd_zero_mean**2, axis=0, keepdims=True)
    # print alpha
    # p_lnd[part_idx, :] = part_lnd_zero_mean*alpha + beta

    # perturb jaw
    part_idx = jaw_idx
    part_lnd = lnd_adversarial[part_idx, :]
    part_lnd_mean = np.mean(part_lnd, 0, keepdims=True)
    part_f = np.sign(lnd_adversarial[part_idx, :]) * epsilon
    part_f_mean = np.mean(part_f, axis=0, keepdims=True)
    beta = part_f_mean + part_lnd_mean
    part_lnd_zero_mean = part_lnd - part_lnd_mean
    alpha = np.mean(part_lnd_zero_mean * (part_lnd + part_f), axis=0, keepdims=True) / np.mean(part_lnd_zero_mean ** 2,
                                                                                               axis=0, keepdims=True)
    alpha = np.clip(alpha, a_min=0.95, a_max=1. / 0.95)
    lnd_adversarial[part_idx, :] = part_lnd_zero_mean * alpha + beta
    return lnd_adversarial