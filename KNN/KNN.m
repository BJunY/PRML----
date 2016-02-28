%************************************************************
%               KNN for regression or classification
%*************************************************************
% the specified parameters are as follows:
%          X: the input of train datas, it should be n*m Matrix(n is the
%             nums of the data, while m is the dimensionalities)
%          y: the output of train datas, t should be n*1 Matrix(n is the
%             nums of the data)
%          k: top k nearest neighbors
%  predict_x: test sample, t should be m*1 Matrix(m is the
%             dimensionalities)
% regression: 1 denotes regression, 0 denotes classification
%
% Author: Bai Junyang
%  Email: bjyhappy123@gmail.com
%************************************************************
function result = KNN(X, y, k, predict_x, regression)
[n, m] = size(X);

%compute the vector of the distanc
predict_X = repmat(predict_x', n, 1);
%size(X)
%size(predict_X)
distance = sum((X - predict_X).^2, 2);

%find the top-K index:topIndex
topIndex = zeros(k, 1);
sort_distance = sort(distance);
for i = 1:k
        topIndex(i) = find(distance == sort_distance(i));
end;

%compute the result
result = mean(y(topIndex)); 
if regression == 0
    if result > 0.5
        result = 1;
    else
        result = 0;
    end;
end;


%plot the point
index = 1:n;
index(topIndex) = [];

if regression == 1
    %plot the predict point
    plot(predict_x, result, 'ro', 'MarkerSize', 10);
    hold on;

    %plot the training data except the top-k data
    for i = index
        plot(X(i,:), y(i), 'ko', 'MarkerSize', 5);
    end;

    %plot the top-k data
    for i = 1:k
        plot(X(topIndex(i),:), y(topIndex(i),:), 'bo', 'MarkerSize', 10);
    end;
else
    %plot the predict point
    if result == 0
        plot(predict_x(1), predict_x(2), 'ro', 'MarkerSize', 10);
        hold on;
    else
        plot(predict_x(1), predict_x(2), 'r+', 'MarkerSize', 10);
        hold on;
    
    end;
    
    %plot the training data except the top-k data
    for i = index
        if y(i) == 0
            plot(X(i, 1), X(i, 2), 'yo', 'MarkerSize', 5);
        else
            plot(X(i, 1), X(i, 2), 'k+', 'MarkerSize', 5);
        end;
    end;
    
    %plot the top-k data
    for i = 1:k
        if y(i) == 0
            plot(X(topIndex(i),1), X(topIndex(i),2), 'bo', 'MarkerSize', 10);
        else
            plot(X(topIndex(i),1), X(topIndex(i),2), 'b+', 'MarkerSize', 10);
        end;
        
    end;
    hold off;
end
