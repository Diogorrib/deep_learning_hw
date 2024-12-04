#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1 (a)
        y_i_hat = np.argmax(self.W.dot(x_i))
        if y_i_hat != y_i:
            self.W[y_i, :] += x_i
            self.W[y_i_hat, :] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.2 (a, b)
        scores = np.expand_dims(self.W.dot(x_i), axis = 1)

        y_one_hot = np.zeros((np.size(self.W, 0),1))
        y_one_hot[y_i] = 1

        exp_scores = np.exp(scores - np.max(scores))
        softmax_probs = exp_scores / np.sum(exp_scores)

        loss_w = (softmax_probs - y_one_hot).dot(np.expand_dims(x_i, axis = 1).T)
        l2_reg_term = l2_penalty * self.W

        gradient = loss_w + l2_reg_term
        self.W -= learning_rate * gradient


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        mean, std = 0.1, 0.1
        self.n_classes = n_classes

        self.W1 = np.random.normal(mean, std, (hidden_size, n_features)) #input-hidden
        self.b1 = np.zeros((hidden_size))

        self.W2 = np.random.normal(mean, std, (n_classes, hidden_size)) #hidden-output
        self.b2 = np.zeros((n_classes))

    def forward_propagation(self, X, expand):
        # ReLU activation
        z1 = self.W1.dot(X.T) + (np.expand_dims(self.b1, axis = 1) if expand else self.b1)
        h1 = np.maximum(0, z1)

        # softmax activation
        z2 = self.W2.dot(h1) + (np.expand_dims(self.b2, axis = 1) if expand else self.b2)
        aux = np.exp(z2 - np.max(z2))
        y_hat = aux / np.sum(aux)

        return z1, h1, y_hat

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.

        _, _, y_hat = self.forward_propagation(X, expand=True)

        predicted_labels = np.argmax(y_hat, axis=0)  # For multi-class classification

        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible
    
    def gradient_and_adjust_dims(self, loss_gradient, h):
        adjs_loss_gradient = np.expand_dims(loss_gradient, axis=1)
        adjs_h = np.expand_dims(h, axis=1)
        return adjs_loss_gradient.dot(adjs_h.T)

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Dont forget to return the loss of the epoch.
        """
        loss = 0
        for x_i, y_i in zip(X, y):
            # Forward Propagation
            z1, h1, y_hat = self.forward_propagation(x_i, False)

            y_one_hot = np.zeros((self.n_classes,))
            y_one_hot[y_i] = 1

            # Calculate the loss
            loss += -y_one_hot.dot(np.log(y_hat+1e-15)) # add 1e-15 to avoid log(0)

            # Backpropagation
            loss_gradient_output = y_hat - y_one_hot # cross-entropy derivative

            gradient_w2 = self.gradient_and_adjust_dims(loss_gradient_output, h1)
            gradient_b2 = loss_gradient_output

            gradient_below = self.W2.T.dot(loss_gradient_output)
            gradient_below_before_activation = np.multiply(gradient_below, np.where(z1 > 0, 1, 0)) # ReLU derivative

            gradient_w1 = self.gradient_and_adjust_dims(gradient_below_before_activation, x_i)
            gradient_b1 = gradient_below_before_activation

            # Updates
            self.W1 -= learning_rate * gradient_w1
            self.b1 -= learning_rate * gradient_b1
            self.W2 -= learning_rate * gradient_w2
            self.b2 -= learning_rate * gradient_b2
        return loss


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
