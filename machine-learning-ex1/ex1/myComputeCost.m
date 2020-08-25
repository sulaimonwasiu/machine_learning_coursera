% Compute Cost
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
%theta = [-1;2];
J = 0;
for i = 1:m
	J = J + (1/(2*m))*((theta'*X(i,:)' - y(i)).^2);
end
% J = (1/(2*m))*sum((theta'*X - y).^2);