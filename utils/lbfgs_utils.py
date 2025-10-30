# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

# coding: utf-8
import numpy as np
import torch


def lbfgs(delta_w, delta_g, v):
    # Concatenar S_k_list e Y_k_list_tensor
    curr_w_k = torch.cat(delta_w, dim=1)
    curr_g_k = torch.cat(delta_g, dim=1)

    a = torch.matmul(curr_w_k.T, curr_g_k)
    d = torch.diag_embed(torch.diag(a))
    low = torch.tril(a, diagonal=-1)
    sigma = torch.matmul(delta_g[-1].T, delta_w[-1]) / torch.matmul(delta_w[-1].T, delta_w[-1])

    upper_mat = torch.cat([-d, low.T], dim=1)
    lower_mat = torch.cat([low, torch.matmul((sigma * curr_w_k.t()), curr_w_k)], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    mat1 = torch.inverse(mat) if torch.det(mat) != 0 else torch.linalg.pinv(mat)
    mat2 = torch.cat([torch.matmul(curr_g_k.T, v), torch.matmul(sigma * curr_w_k.T, v)], dim=0)
    p = torch.matmul(mat1, mat2)

    return sigma * v - torch.matmul(torch.cat([curr_g_k, sigma * curr_w_k], dim=1), p)


def calculo_FLDet(e, global_net, grad_list, last_weight, old_grad_list, weight_record, grad_record, c):
    param_list = [torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in grad_list]
    grad = torch.mean(torch.cat(param_list, dim=1), dim=-1, keepdim=True)
    tmp = [param.data.clone() for param in global_net.parameters()]
    weight = torch.cat([xx.view(-1, 1) for xx in tmp], dim=0)
    norm_distance = None
    if e < c.FLDET_START:
        trust_scores = [1 / len(grad_list)] * len(grad_list)
    else:
        hvp = lbfgs(weight_record, grad_record, weight - last_weight)
        norm_distance, distance = calculate_score(old_grad_list, param_list, hvp)
        trust_scores = distance_to_score(distance)
    return norm_distance, trust_scores, param_list, weight, grad


# -- FUNCIÓNS AUXILIARES -- #
def calculate_score(old_gradients, param_list, hvp=None):
    pred_grad = []

    # Predición dos gradientes --> gradiente vello + hvp
    for i in range(len(old_gradients)):
        pred_grad.append(old_gradients[i] + hvp)

    distancia = torch.norm(torch.cat(pred_grad, dim=1) - torch.cat(param_list, dim=1), dim=0).cpu().numpy()

    return distancia / np.sum(distancia), distancia


def distance_to_score(distances, temperature=1.0):
    scores = np.array(distances)

    # Invertir sospecha → confianza base
    inv_scores = -scores  # menor sospecha → valor mayor

    # Aplicar softmax con temperatura (ajusta la sensibilidad)
    exp_scores = np.exp(inv_scores / temperature)
    return exp_scores / np.sum(exp_scores)
