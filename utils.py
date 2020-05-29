import argparse
import math
import numpy
import os
import pdb
import re
from datetime import datetime
from os import path

import sklearn.manifold as manifold
import torch
from PIL import Image, ImageDraw
from sklearn import decomposition




def format_losses(loss_i, loss_s, loss_p=None, split="train"):
    log_string = " "
    log_string += f"{split} loss ["
    log_string += f"i: {loss_i:.5f}, "
    log_string += f"s: {loss_s:.5f}, "
    if loss_p is not None:
        log_string += f", p: {loss_p:.5f}"
    log_string += "]"
    return log_string


def save_movie(
    dirname,
    images,
    states,
    costs=None,
    actions=None,
    mu=None,
    std=None,
    pytorch=True,
    raw=False,
):
    images = images.data if hasattr(images, "data") else images
    states = states.data if hasattr(states, "data") else states
    if costs is not None:
        costs = costs.data if hasattr(costs, "data") else costs
    if actions is not None:
        actions = actions.data if hasattr(actions, "data") else actions

    os.system("mkdir -p " + dirname)
    print(f"[saving movie to {dirname}]")
    if mu is not None:
        mu = mu.squeeze()
        std = std.squeeze()
    else:
        mu = actions
    if pytorch:
        images = images.permute(0, 2, 3, 1).cpu().numpy() * 255
    if raw:
        for t in range(images.shape[0]):
            img = images[t]
            img = numpy.uint8(img)
            Image.fromarray(img).save(path.join(dirname, f"im{t:05d}.png"))
        return
    for t in range(images.shape[0]):
        img = images[t]
        img = numpy.concatenate(
            (img, numpy.zeros((24, 24, 3)).astype("float")), axis=0
        )
        img = numpy.uint8(img)
        pil = Image.fromarray(img).resize(
            (img.shape[1] * 5, img.shape[0] * 5), Image.NEAREST
        )
        draw = ImageDraw.Draw(pil)

        text = ""
        if states is not None:
            text += f"x: [{states[t][0]:.2f}, {states[t][1]:.2f} \n"
            text += f"dx: {states[t][2]:.2f}, {states[t][3]:.2f}]\n"
        if costs is not None:
            text += f"c: [{costs[t][0]:.2f}, {costs[t][1]:.2f}]\n"
        if actions is not None:
            text += f"a: [{actions[t][0]:.2f}, {actions[t][1]:.2f}]\n"
            x = int(images[t].shape[1] * 5 / 2 - mu[t][1] * 30)
            y = int(images[t].shape[0] * 5 / 2 - mu[t][0] * 30)
            if std is not None:
                ex = max(3, int(std[t][1] * 100))
                ey = max(3, int(std[t][0] * 100))
            else:
                ex, ey = 3, 3
            bbox = (x - ex, y - ey, x + ex, y + ey)
            draw.ellipse(bbox, fill=(200, 200, 200))

        draw.text((10, 130 * 5 - 10), text, (255, 255, 255))
        pil.save(dirname + f"/im{t:05d}.png")


def grad_norm(net):
    total_norm = 0
    for p in net.parameters():
        if p.grad is None:
            pdb.set_trace()
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def create_tensorboard_writer(opt):
    return None
