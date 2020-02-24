% Ramya Muthukrishnan, ENGR105, Spring 2019, HW 11
% Collaborators: None
% neuralNetwork.m
% This program trains a neural network to recognize 4 letters: A, C, G, and
% T. It displays the learned weight matrices from the network and displays
% the output when these matrices are applied to various inputs.

% define initial weight matrices
rng(0)
V0 = randn(25,25);
W0 = randn(2,25);

% train the neural network
[V, W] = nnTrain(V0,W0,10000,0.1);

% plot image of V
subplot(2,2,1), imagesc(V)
colormap jet
colorbar
title('Trained V')
axis equal
axis off

% plot histogram of V
subplot(2,2,2), histogram(V, 20)
title('Distribution of trained V values')

% plot image of W
subplot(2,2,3), imagesc(W)
colormap jet
colorbar
title('Trained W')
axis equal
axis off

% plot histogram of W
subplot(2,2,4), histogram(W, 20)
title('Distribution of trained W values')

% create matrix of 12 input vectors: 1st 4 rows
A = [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1];
C = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1];
G = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1];
T = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0];
inp_mat = [A; C; G; T; A; C; G; T; A; C; G; T];

% next 4 rows
for i=5:8
    r = randi(25);
    if inp_mat(i,r) == 0
        inp_mat(i,r) = 1;
    else
        inp_mat(i,r) = 0;
    end
end

% last 4 rows
for i=9:12
    r = randperm(25,2);
    if inp_mat(i,r(1)) == 0
        inp_mat(i,r(1)) = 1;
    else
        inp_mat(i,r(1)) = 0;
    end
    if inp_mat(i,r(2)) == 0
        inp_mat(i,r(2)) = 1;
    else
        inp_mat(i,r(2)) = 0;
    end
end

% apply learned V and W to input matrix
inp_mat = inp_mat';
output_mat = sigmoid(W*sigmoid(V*inp_mat));
% round output matrix so that it consists of just 0s and 1s
output_mat = round(output_mat);
output_mat = output_mat';
disp("Outputs:")
disp(output_mat)

% plot each input vector using the plot_letters function
figure()
for i = 1:12
    subplot(3,4,i), plot_letters(inp_mat(:,i))
end

% display sequence of letters represented by input matrix
seq = '';
for i = 1:12
    if output_mat(i,:)==[0, 0]
        new = 'T';
    elseif output_mat(i,:)==[0, 1]
        new = 'A';
    elseif output_mat(i,:)==[1, 0]
        new = 'C';
    else
        new = 'G';
    end
    seq = strcat(seq,new);
end
disp(['Sequence: ', seq])

%% sigmoid.m
% <include>sigmoid.m</include>
%% plot_letters.m
% <include>plot_letters.m</include>
%% nnTrain.m
% <include>nnTrain.m</include>