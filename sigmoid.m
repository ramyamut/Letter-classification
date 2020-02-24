% Ramya Muthukrishnan, ENGR105, Spring 2019, HW 11
% Collaborators: None
% function s = sigmoid(x)
% Passes each element of a vector through the sigmoid function, 1/(1 +
% e^(0.5-x))
% Inputs: x = input vector
% Outputs: s = output vector, where each corresponding element o x is equal
% to 1/(1 + e^(0.5-x))
% Usage example: if x = [0 1 2], the function call s = sigmoid(x) returns
% the vector [1/(1 + e^(0.5-0)), 1/(1 + e^(0.5-1)), 1/(1 + e^(0.5-2))]

function s = sigmoid(x)
    s = 1./(1+exp(0.5-x));
end