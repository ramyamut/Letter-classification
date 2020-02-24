% Ramya Muthukrishnan, ENGR105, Spring 2019, HW 11
% Collaborators: None
% function [V, W] = nnTrain(V0,W0,max_iter,rate)
% Trains a neural network by taking in 2 initial weight matrices, a maximum
% number of optimization steps, and the learning rate, and outputting the
% optimized weight matrices
% Inputs: V0 = 25x25 matrix containing the initial weights for the
% dependencies between the input layer and the hidden layer
% W0 = 2x25 matrix containing the initial weights for the dependencies
% between the hidden layer and the output layer
% max_iter = maximum number of steps in the optimization
% rate = the learning rate of the optimization
% Outputs: V = 25x25 matrix containing the learned weights for the
% dependencies between the input layer and the hidden layer
% W = 2x25 matrix containing the learned weights for the dependencies
% between the hidden layer and the output layer
% Usage example: if V0 = randn(25,25) and W0 = randn(2,25), then the
% function call [V, W] = nnTrain(V0,W0,10000,0.1) returns the learned
% weight matrices for the network, where the number of iterations is 10000,
% the learning rate is 0.1, and the initial weight matrices consist of
% values from a normal distribution.

function [V, W]= nnTrain(V0,W0,max_iter,rate)
    % define inputs for each letter
    in_T = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0];
    in_A = [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1];
    in_C = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1];
    in_G = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1];
    in_mat = [in_T; in_A; in_C; in_G];
    % define targets for each letter
    tar_T = [0, 0];
    tar_A = [0, 1];
    tar_C = [1, 0];
    tar_G = [1, 1];
    tar_mat = [tar_T; tar_A; tar_C; tar_G];
    % initialize V and W
    V = V0;
    W = W0;
    % loop through all iterations
    for iter=1:max_iter
        % randomly select an input vector to train on
        r1 = randi(4);
        in = (in_mat(r1,:))';
        tar = (tar_mat(r1,:))';
        % randomly choose a row of W to update
        r2 = randi(2);
        w = W(r2,:);
        % compute output and hidden layer vectors
        q = sigmoid(V*in);
        out = sigmoid(W*q);
        % compute gradients
        V_grad = (out(r2)-tar(r2))*out(r2)*(1-out(r2))*(w'.*q.*(1-q))*in';
        W_grad = (out(r2)-tar(r2))*out(r2)*(1-out(r2))*q';
        % update V and W
        V = V - rate*V_grad;
        W(r2,:) = W(r2,:) - rate*W_grad;
    end
end