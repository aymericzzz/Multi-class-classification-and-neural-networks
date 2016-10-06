function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% +1 de l'input layer
X = [ones(m,1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
temp = zeros(num_labels,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% on déclare le hidden layer et le ouput layer
a2 = zeros(5000, 25);
a3 = zeros(m, num_labels);

% computation du hidden layer avec les inputs et Theta1 qui est donné
a2 = sigmoid(X*Theta1');

% on rajoute une colonne de 1 au hidden layer (+1)
a2 = [ones(size(a2,1),1) a2];

% computation du output layer avec Theta2 et le hidden layer. On obtient une matrice 5000x10, où chaque i,j représente la probabilité de l'input i
% à correspondre au label j.
a3 = sigmoid(a2*Theta2');

% on place dans p, l'indice (=> le chiffe auquel correspond l'input) de la "meilleure probabilité"
[~, p] = max(a3, [], 2);	







% =========================================================================


end
